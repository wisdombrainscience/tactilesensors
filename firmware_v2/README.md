# Firmware v2 (grasp controller + packed frames)

`firmware_v2.patch` applies to the original STM32CubeIDE project
`Code MCU (STM32CubeIDE)/05_FreeRTOSv2_Velostat_Queue_FingerControl`
(`patch -p1 --binary` from the project's parent folder, or apply the hunks by hand).
Copy `../firmware_patch/*.c|h` into a new `GraspCtrl/` folder in the project.

Build switches (CubeIDE: *Project > Properties > C/C++ Build > Settings > Preprocessor*):
* `GRASP_CTRL_V2`: timers at 1 µs/count (prescaler 72, period 20000), 16-bit servo limits,
  GUI limits/steps scaled by `PWM_SCALE` = 20, and per-finger conductance-domain PI in `StartHandPeakControl`.
* `TF_PACKED_FRAMES`: 665-byte packed frame with sequence number and CRC-16 at 921.6 kbaud.
  Use `code/host/tactile_logger.py`; the original C# GUI only reads legacy frames.
  Without the switch, the legacy 884-byte frame is sent.

## Verified here (no hardware)
* Cross-compiles with arm-none-eabi-gcc 13.2 (`make -f build.mk EXTRA="-DGRASP_CTRL_V2 -DTF_PACKED_FRAMES" EXTRAINC=-IGraspCtrl EXTRASRC="GraspCtrl/grasp_ctrl.c GraspCtrl/tactile_frame.c"`).
  * Flash: 29.1 KiB of 64 KiB.
  * RAM: 19.6 KiB of 20 KiB. That leaves 392 B above the linker's reserved 1.5 KiB of stack and heap. Keep an eye on this margin.
  * The build without the switches also compiles (28.1 KiB flash), so the original behaviour is preserved.
* `fw_v2.elf` and `fw_v2.bin` are the cross-compiled image. It has **not** been flashed or tested on the hand.
* Software-in-the-loop (`code/sil/sil_test.py`): the C controller matches the Python reference (median error 0.08–0.09 %, settling 0.42 s compliant / 0.12 s stiff), including reversed joints.
* Instruction counts on an emulated Cortex-M3 (unicorn):
  * `grasp_ctrl_update`: at most 1,420 instructions per call.
  * Packing a frame: 11.7k instructions with the table-driven CRC.
