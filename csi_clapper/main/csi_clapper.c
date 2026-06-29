/*
 * CSI Clapperboard — spare ESP32-C6 acts as a sync slate between CSI capture
 * and camera video. Press the BOOT button to send a UDP "clap" packet to the
 * laptop (alternating START/STOP events) and flash the on-board WS2812 LED
 * with a bright color visible to the overhead camera.
 *
 * - Connects to the CSI_TX Soft-AP as a STA
 * - On button press: UNICASTS a short burst of identical 8-byte clapper packets
 *   to the laptop on UDP :5500 (csi_collector.py distinguishes them from CSI by
 *   magic byte 0xCA and de-dups the burst by (event, seq)). The burst makes a
 *   single UDP loss harmless — a dropped clap would otherwise lose a session
 *   boundary. Unicast (not broadcast) avoids the AP's DTIM beacon buffering.
 * - LED: WS2812 addressable (uses led_strip driver). GPIO is target-dependent:
 *   GPIO8 on C6 dev kit, GPIO48 on diymore S3-WROOM-1.
 * - Button: BOOT button, active LOW. GPIO9 on C6, GPIO0 on S3.
 *
 * Press sequence: 1st=START, 2nd=STOP, 3rd=START, 4th=STOP, ...
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "driver/gpio.h"
#include "lwip/sockets.h"
#include "led_strip.h"

static const char *TAG = "CSI_CLAPPER";

#define CSI_TX_SSID   "CSI_TX"
#define CSI_TX_PASS   "23456789"

#define COLLECTOR_PORT  5500
#define LAPTOP_IP       "192.168.4.200"   /* laptop's static IP on CSI_TX */
#define CLAP_BURST      5                 /* identical packets per press (loss tolerance) */
#if CONFIG_IDF_TARGET_ESP32S3
#define LED_GPIO        GPIO_NUM_48      /* WS2812 on diymore S3-WROOM-1 */
#define BUTTON_GPIO     GPIO_NUM_0       /* BOOT button on S3 dev kit */
#else
#define LED_GPIO        GPIO_NUM_8       /* WS2812 on C6 dev kit */
#define BUTTON_GPIO     GPIO_NUM_9       /* BOOT button on C6 dev kit */
#endif
#define DEBOUNCE_MS     300

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

static led_strip_handle_t s_led;
static int s_sock = -1;
static struct sockaddr_in s_dest;

typedef struct __attribute__((packed)) {
    uint8_t  magic;          /* 0xCA */
    uint8_t  event;          /* 0=start, 1=stop, 2=clap */
    uint32_t timestamp_us;
    uint16_t seq;
} clapper_packet_t;

#define CLAP_MAGIC      0xCA
#define EVENT_START     0
#define EVENT_STOP      1

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
        case WIFI_EVENT_STA_START:
            esp_wifi_connect();
            break;
        case WIFI_EVENT_STA_DISCONNECTED:
            ESP_LOGW(TAG, "Disconnected, reconnecting...");
            xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
            esp_wifi_connect();
            break;
        default:
            break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void led_set_color(uint8_t r, uint8_t g, uint8_t b)
{
    led_strip_set_pixel(s_led, 0, r, g, b);
    led_strip_refresh(s_led);
}

static void led_off(void)
{
    led_strip_clear(s_led);
}

static void led_flash_event(int event)
{
    if (event == EVENT_START) {
        led_set_color(0, 255, 64);     /* bright green */
    } else {
        led_set_color(255, 32, 0);     /* bright red */
    }
    vTaskDelay(pdMS_TO_TICKS(1500));   /* hold ~45 frames at 30fps */
    led_off();
}

static void send_clap(int event, uint16_t seq)
{
    if (s_sock < 0) {
        ESP_LOGW(TAG, "Socket not ready, can't send");
        return;
    }
    clapper_packet_t pkt = {
        .magic = CLAP_MAGIC,
        .event = (uint8_t)event,
        .timestamp_us = (uint32_t)esp_timer_get_time(),
        .seq = seq,
    };
    /* Burst: send CLAP_BURST identical packets (same event+seq) ~5 ms apart so a
     * single UDP loss can't drop a session boundary. Collector de-dups by seq. */
    int ok = 0;
    for (int i = 0; i < CLAP_BURST; i++) {
        if (sendto(s_sock, &pkt, sizeof(pkt), 0,
                   (struct sockaddr *)&s_dest, sizeof(s_dest)) >= 0) {
            ok++;
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    if (ok == 0) {
        ESP_LOGE(TAG, "sendto failed for all %d in burst: errno %d", CLAP_BURST, errno);
    } else {
        ESP_LOGI(TAG, "CLAP sent: event=%s seq=%u (%d/%d delivered)",
                 event == EVENT_START ? "START" : "STOP", seq, ok, CLAP_BURST);
    }
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
    led_off();
}

static void init_button(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
}

static void init_wifi_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

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

static void init_socket(void)
{
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        return;
    }
    memset(&s_dest, 0, sizeof(s_dest));
    s_dest.sin_family = AF_INET;
    s_dest.sin_addr.s_addr = inet_addr(LAPTOP_IP);   /* unicast, like the RX boards */
    s_dest.sin_port = htons(COLLECTOR_PORT);

    ESP_LOGI(TAG, "Socket ready (UDP unicast to %s:%d)", LAPTOP_IP, COLLECTOR_PORT);
}

static void init_nvs(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
}

void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  CSI CLAPPERBOARD");
    ESP_LOGI(TAG, "  Press BOOT button to clap.");
    ESP_LOGI(TAG, "  1st = START, 2nd = STOP, 3rd = START, ...");
    ESP_LOGI(TAG, "========================================");

    init_nvs();
    init_led();
    init_button();

    /* Brief blue pulse so we know firmware is alive before WiFi connects */
    led_set_color(0, 0, 64);
    vTaskDelay(pdMS_TO_TICKS(300));
    led_off();

    init_wifi_sta();

    ESP_LOGI(TAG, "Waiting for %s...", CSI_TX_SSID);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                        pdFALSE, pdTRUE, portMAX_DELAY);

    init_socket();

    /* Triple cyan blink = ready */
    for (int i = 0; i < 3; i++) {
        led_set_color(0, 255, 255);
        vTaskDelay(pdMS_TO_TICKS(120));
        led_off();
        vTaskDelay(pdMS_TO_TICKS(120));
    }

    ESP_LOGI(TAG, "Ready. Press BOOT to clap.");

    uint16_t seq = 0;
    int next_event = EVENT_START;
    uint32_t last_press_ms = 0;

    while (1) {
        int level = gpio_get_level(BUTTON_GPIO);
        if (level == 0) {  /* button pressed, active LOW */
            uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
            if (now_ms - last_press_ms > DEBOUNCE_MS) {
                last_press_ms = now_ms;
                seq++;
                send_clap(next_event, seq);
                led_flash_event(next_event);
                next_event = (next_event == EVENT_START) ? EVENT_STOP : EVENT_START;
            }
            /* Wait for release */
            while (gpio_get_level(BUTTON_GPIO) == 0) {
                vTaskDelay(pdMS_TO_TICKS(20));
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
