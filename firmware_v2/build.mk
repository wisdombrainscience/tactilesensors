# Out-of-IDE build of the STM32CubeIDE project (arm-none-eabi-gcc)
CC=arm-none-eabi-gcc
CFLAGS=-mcpu=cortex-m3 -mthumb -std=gnu11 -Os -ffunction-sections -fdata-sections -Wall -DUSE_HAL_DRIVER -DSTM32F103xB $(EXTRA)
INC=-ICore/Inc -IDrivers/STM32F1xx_HAL_Driver/Inc -IDrivers/STM32F1xx_HAL_Driver/Inc/Legacy -IDrivers/CMSIS/Device/ST/STM32F1xx/Include -IDrivers/CMSIS/Include -IMiddlewares/Third_Party/FreeRTOS/Source/include -IMiddlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS_V2 -IMiddlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM3 -IMyArray -IScan_74238 -IServo/Inc $(EXTRAINC)
SRC=$(shell find Core/Src Drivers/STM32F1xx_HAL_Driver/Src Middlewares MyArray Scan_74238 Servo/Src -name '*.c') $(EXTRASRC)
OUT?=build
$(OUT)/fw.elf: $(SRC) Core/Startup/startup_stm32f103c8tx.s
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INC) $(SRC) Core/Startup/startup_stm32f103c8tx.s -TSTM32F103C8TX_FLASH.ld -Wl,--gc-sections -Wl,-Map=$(OUT)/fw.map --specs=nano.specs --specs=nosys.specs -lc -lm -o $@
	arm-none-eabi-size $@
