#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_console.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "driver/gpio.h"
#include "led_strip.h"
#include "lwip/sockets.h"

static const char *TAG = "CSI_RX";

#define CSI_TX_SSID   "CSI_TX"
#define CSI_TX_PASS   "23456789"

/* AP gateway IP */
#define AP_IP         "192.168.4.1"
#define PING_PORT     1234
/* CSI sample rate is driven by how often each RX pings the AP (CSI is extracted
 * from the replies). Override at flash time without editing source:
 *     idf.py -DPING_INTERVAL_MS=25 flash      (~40 CSI/sec)
 * Aim to match the camera framerate (~30/sec) so every camera frame gets a
 * fresh, unique CSI reading (no duplicate-CSI rows). Falls back to 100 (~10/s). */
#ifndef PING_INTERVAL_MS
#define PING_INTERVAL_MS  100
#endif

/* CSI data forwarding — UDP UNICAST to the laptop collector (port 5500).
 * Broadcast was buffered by the AP until each DTIM beacon (~102 ms), capping
 * delivery at ~10 Hz; unicast is relayed promptly for the full ~30 Hz.
 *
 * BOARD_ID and LAPTOP_IP are now stored in NVS (namespace "cfg") so ONE firmware
 * image serves every board — set them once over the USB serial console:
 *     setid 1            (or 4, or 5)
 *     setip 192.168.4.200
 *     cfg                (show current)
 * The compile-time values below are only FALLBACK DEFAULTS used when NVS is
 * empty (e.g. a freshly-erased board). The old `-DBOARD_ID=` / `-DLAPTOP_IP=`
 * flags still work as a way to bake a default in. */
#define COLLECTOR_PORT    5500
#ifndef LAPTOP_IP
#define LAPTOP_IP         "192.168.4.200"
#endif
#ifndef BOARD_ID
#define BOARD_ID          3
#endif

/* Status LED (on-board WS2812). GPIO is target-dependent. */
#if CONFIG_IDF_TARGET_ESP32S3
#define LED_GPIO          GPIO_NUM_48
#else
#define LED_GPIO          GPIO_NUM_8        /* C6 dev kit */
#endif

/* Event group bits */
#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

/* Runtime config (loaded from NVS, falling back to the #defines above) */
static uint8_t g_board_id;
static char    g_laptop_ip[16];

static uint32_t s_csi_count = 0;
static uint32_t s_udp_send_count = 0;
static int s_collector_sock = -1;
static struct sockaddr_in s_collector_addr;
static led_strip_handle_t s_led = NULL;

typedef enum { LED_CONNECTING, LED_CONNECTED, LED_DISCONNECTED } led_state_t;
static volatile led_state_t s_led_state = LED_CONNECTING;

/* ── UDP CSI packet format (145 bytes on the wire) ───────────────── */

typedef struct __attribute__((packed)) {
    uint8_t  board_id;
    uint8_t  mac[6];
    int8_t   rssi;
    uint8_t  channel;
    uint32_t timestamp_us;
    uint16_t rx_seq;
    uint16_t csi_len;
    int8_t   csi_data[128];
} csi_udp_packet_t;  /* 1+6+1+1+4+2+2+128 = 145 bytes */

/* ── Status LED ──────────────────────────────────────────────────── */

static void led_set(uint8_t r, uint8_t g, uint8_t b)
{
    if (!s_led) return;
    led_strip_set_pixel(s_led, 0, r, g, b);
    led_strip_refresh(s_led);
}

static void init_led(void)
{
    led_strip_config_t strip_cfg = {
        .strip_gpio_num = LED_GPIO,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt_cfg = {
        .resolution_hz = 10 * 1000 * 1000,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_cfg, &rmt_cfg, &s_led));
    led_strip_clear(s_led);
}

/* The ONLY place that touches the LED hardware (avoids cross-task RMT races):
 *   yellow = connecting   blue = connected + CSI streaming
 *   purple = connected but CSI stalled   red = link dropped */
static void led_task(void *arg)
{
    uint32_t prev = 0;
    while (1) {
        switch (s_led_state) {
        case LED_DISCONNECTED:
            led_set(64, 0, 0);                       /* red */
            break;
        case LED_CONNECTING:
            led_set(64, 40, 0);                      /* yellow */
            break;
        case LED_CONNECTED: {
            uint32_t now = s_csi_count;
            if (now != prev) led_set(0, 0, 64);      /* blue: CSI flowing */
            else             led_set(48, 0, 48);     /* purple: connected, no CSI */
            prev = now;
            break;
        }
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ── WiFi event handler ──────────────────────────────────────────── */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
        case WIFI_EVENT_STA_START:
            s_led_state = LED_CONNECTING;
            esp_wifi_connect();
            break;
        case WIFI_EVENT_STA_DISCONNECTED:
            ESP_LOGW(TAG, "Disconnected from AP, reconnecting...");
            s_led_state = LED_DISCONNECTED;
            xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
            esp_wifi_connect();
            break;
        default:
            break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_led_state = LED_CONNECTED;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ── CSI callback ────────────────────────────────────────────────── */

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info)
{
    if (info == NULL || s_collector_sock < 0) {
        return;
    }

    s_csi_count++;

    csi_udp_packet_t pkt = {0};
    pkt.board_id = g_board_id;
    memcpy(pkt.mac, info->mac, 6);
    pkt.rssi = info->rx_ctrl.rssi;
    pkt.channel = info->rx_ctrl.channel;
    pkt.timestamp_us = info->rx_ctrl.timestamp;
    pkt.rx_seq = info->rx_seq;

    int start = info->first_word_invalid ? 4 : 0;
    int data_len = info->len - start;
    if (data_len > 128) data_len = 128;
    if (data_len < 0) data_len = 0;
    pkt.csi_len = (uint16_t)data_len;
    memcpy(pkt.csi_data, info->buf + start, data_len);

    /* Header is fixed 17 bytes; send header + exactly csi_len CSI bytes. */
    size_t send_len = 17 + (size_t)data_len;
    int sent = sendto(s_collector_sock, &pkt, send_len, 0,
                      (struct sockaddr *)&s_collector_addr, sizeof(s_collector_addr));
    if (sent > 0) {
        s_udp_send_count++;
    }
}

/* ── Traffic generator task ──────────────────────────────────────── */

static void traffic_gen_task(void *pvParameters)
{
    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(AP_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PING_PORT);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Traffic generator started: UDP to %s:%d every %dms",
             AP_IP, PING_PORT, PING_INTERVAL_MS);

    uint32_t seq = 0;
    char payload[32];

    while (1) {
        seq++;
        snprintf(payload, sizeof(payload), "csi_ping_%lu", (unsigned long)seq);
        sendto(sock, payload, strlen(payload), 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr));

        vTaskDelay(pdMS_TO_TICKS(PING_INTERVAL_MS));
    }
}

/* ── Collector socket init ───────────────────────────────────────── */

static void init_collector_socket(void)
{
    s_collector_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_collector_sock < 0) {
        ESP_LOGE(TAG, "Failed to create collector socket: errno %d", errno);
        return;
    }

    memset(&s_collector_addr, 0, sizeof(s_collector_addr));
    s_collector_addr.sin_family = AF_INET;
    s_collector_addr.sin_addr.s_addr = inet_addr(g_laptop_ip);
    s_collector_addr.sin_port = htons(COLLECTOR_PORT);

    ESP_LOGI(TAG, "Collector socket ready — unicasting CSI to %s:%d", g_laptop_ip, COLLECTOR_PORT);
}

/* ── NVS-backed config ───────────────────────────────────────────── */

static void load_config(void)
{
    /* defaults first */
    g_board_id = BOARD_ID;
    strncpy(g_laptop_ip, LAPTOP_IP, sizeof(g_laptop_ip) - 1);
    g_laptop_ip[sizeof(g_laptop_ip) - 1] = '\0';

    nvs_handle_t h;
    if (nvs_open("cfg", NVS_READONLY, &h) == ESP_OK) {
        uint8_t id;
        if (nvs_get_u8(h, "board_id", &id) == ESP_OK) {
            g_board_id = id;
        }
        size_t len = sizeof(g_laptop_ip);
        /* if the key is absent, nvs_get_str leaves g_laptop_ip at its default */
        nvs_get_str(h, "laptop_ip", g_laptop_ip, &len);
        nvs_close(h);
    }
    ESP_LOGI(TAG, "Config: board_id=%d laptop_ip=%s (set via console: setid/setip)",
             g_board_id, g_laptop_ip);
}

static esp_err_t cfg_set_u8(const char *key, uint8_t v)
{
    nvs_handle_t h;
    esp_err_t e = nvs_open("cfg", NVS_READWRITE, &h);
    if (e != ESP_OK) return e;
    e = nvs_set_u8(h, key, v);
    if (e == ESP_OK) e = nvs_commit(h);
    nvs_close(h);
    return e;
}

static esp_err_t cfg_set_str(const char *key, const char *v)
{
    nvs_handle_t h;
    esp_err_t e = nvs_open("cfg", NVS_READWRITE, &h);
    if (e != ESP_OK) return e;
    e = nvs_set_str(h, key, v);
    if (e == ESP_OK) e = nvs_commit(h);
    nvs_close(h);
    return e;
}

/* ── Console commands (USB serial) ───────────────────────────────── */

static int cmd_setid(int argc, char **argv)
{
    if (argc < 2) { printf("usage: setid <1..200>\n"); return 1; }
    int v = atoi(argv[1]);
    if (v < 1 || v > 200) { printf("error: id must be 1..200\n"); return 1; }
    if (cfg_set_u8("board_id", (uint8_t)v) != ESP_OK) { printf("error: NVS write failed\n"); return 1; }
    g_board_id = (uint8_t)v;                 /* applies to subsequent packets */
    printf("ok: board_id = %d saved\n", v);
    return 0;
}

static int cmd_setip(int argc, char **argv)
{
    if (argc < 2) { printf("usage: setip <a.b.c.d>\n"); return 1; }
    if (inet_addr(argv[1]) == INADDR_NONE) { printf("error: invalid IPv4 address\n"); return 1; }
    if (cfg_set_str("laptop_ip", argv[1]) != ESP_OK) { printf("error: NVS write failed\n"); return 1; }
    strncpy(g_laptop_ip, argv[1], sizeof(g_laptop_ip) - 1);
    g_laptop_ip[sizeof(g_laptop_ip) - 1] = '\0';
    printf("ok: laptop_ip = %s saved (reboot to apply to the socket)\n", argv[1]);
    return 0;
}

static int cmd_cfg(int argc, char **argv)
{
    printf("board_id=%d laptop_ip=%s\n", g_board_id, g_laptop_ip);
    return 0;
}

static void start_console(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "rx>";
    repl_cfg.max_cmdline_length = 64;

    esp_console_dev_usb_serial_jtag_config_t hw = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw, &repl_cfg, &repl));

    const esp_console_cmd_t cmds[] = {
        { .command = "setid", .help = "Set board id (1..200) in NVS", .func = &cmd_setid },
        { .command = "setip", .help = "Set laptop unicast IP in NVS",  .func = &cmd_setip },
        { .command = "cfg",   .help = "Show current board id + laptop ip", .func = &cmd_cfg },
    };
    for (size_t i = 0; i < sizeof(cmds) / sizeof(cmds[0]); i++) {
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
    }
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

/* ── CSI init ────────────────────────────────────────────────────── */

static void init_csi(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "Power save disabled");

    wifi_csi_config_t csi_config = {
        .acquire_csi_legacy = 1,
        .acquire_csi_ht20 = 1,
        .acquire_csi_ht40 = 1,
        .acquire_csi_su = 1,
        .val_scale_cfg = 0,
        .dump_ack_en = 0,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI collection enabled");
}

/* ── WiFi STA init ───────────────────────────────────────────────── */

static void init_wifi_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CSI_TX_SSID,
            .password = CSI_TX_PASS,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi STA started, connecting to %s...", CSI_TX_SSID);
}

/* ── NVS init ────────────────────────────────────────────────────── */

static void init_nvs(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
}

/* ── Main ────────────────────────────────────────────────────────── */

void app_main(void)
{
    init_nvs();
    load_config();

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  CSI RECEIVER — Board #%d", g_board_id);
    ESP_LOGI(TAG, "========================================");

    init_led();
    xTaskCreate(led_task, "led", 2048, NULL, 4, NULL);

    /* Console up before WiFi so a board can be configured even if it can't
     * associate (setid / setip / cfg over USB serial). */
    start_console();

    init_wifi_sta();

    ESP_LOGI(TAG, "Waiting for connection to %s...", CSI_TX_SSID);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                        pdFALSE, pdTRUE, portMAX_DELAY);

    init_collector_socket();
    init_csi();

    /* Start traffic generator — UDP packets trigger CSI on the AP's responses */
    xTaskCreate(traffic_gen_task, "traffic_gen", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "Board #%d sending CSI via UDP unicast to %s:%d",
             g_board_id, g_laptop_ip, COLLECTOR_PORT);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "CSI: %lu received, %lu sent via UDP",
                 (unsigned long)s_csi_count,
                 (unsigned long)s_udp_send_count);
    }
}
