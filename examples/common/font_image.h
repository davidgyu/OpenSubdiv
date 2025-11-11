//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef FONT_IMAGE_H
#define FONT_IMAGE_H

#define FONT_TEXTURE_WIDTH 128
#define FONT_TEXTURE_HEIGHT 128
#define FONT_TEXTURE_COLUMNS 16
#define FONT_TEXTURE_ROWS 8
#define FONT_CHAR_WIDTH (FONT_TEXTURE_WIDTH/FONT_TEXTURE_COLUMNS)
#define FONT_CHAR_HEIGHT (FONT_TEXTURE_HEIGHT/FONT_TEXTURE_ROWS)
#define FONT_CHECK_BOX_OFF     0x2
#define FONT_CHECK_BOX_ON      0x3
#define FONT_RADIO_BUTTON_OFF  0x4
#define FONT_RADIO_BUTTON_ON   0x5
#define FONT_SLIDER_LEFT       0x10
#define FONT_SLIDER_MIDDLE     0x11
#define FONT_SLIDER_RIGHT      0x12
#define FONT_SLIDER_CURSOR     0x13
#define FONT_ARROW_RIGHT       0x14
#define FONT_ARROW_DOWN        0x15

extern unsigned char font_image[];

#endif // FONT_IMAGE_H
