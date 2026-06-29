idf_component_register(SRCS "csi_rx_main.c"
                       INCLUDE_DIRS ".")

# Allow setting these at build time without editing source:
#     idf.py -DBOARD_ID=1 -DPING_INTERVAL_MS=25 flash
if(DEFINED BOARD_ID)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BOARD_ID=${BOARD_ID})
endif()
if(DEFINED PING_INTERVAL_MS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PING_INTERVAL_MS=${PING_INTERVAL_MS})
endif()
