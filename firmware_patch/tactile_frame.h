/**
 * tactile_frame.h -- compact, checksummed tactile frame (replaces the raw
 * 884-byte uint16 frame 0xAAAA | 440 x uint16 | 0x5555).
 *
 *   byte 0..1  : sync 0xA5 0x5A
 *   byte 2     : sequence number (wraps)
 *   byte 3..662: 440 x 12-bit samples, two samples packed in 3 bytes
 *   byte 663..664: CRC-16/CCITT-FALSE over bytes 2..662 (big-endian)
 * Total 665 bytes (-25 %).  Use with USB-CDC or UART >= 921600 baud.
 */
#ifndef TACTILE_FRAME_H
#define TACTILE_FRAME_H
#include <stdint.h>
#define TF_CELLS 440u
#define TF_BYTES (2u + 1u + (TF_CELLS * 3u) / 2u + 2u)
uint16_t tf_build(const uint16_t *cells, uint8_t seq, uint8_t *out);
#endif
