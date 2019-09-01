import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache


def random_choice_along_axis(arr, axis):
    idx = np.random.randint(0, arr.shape[axis])
    return np.take(arr, idx, axis)


class Tile:
    """Methods related to visualising tiles."""

    @staticmethod
    def get_color_code(value: int, as_hex=False, as_rgb=False):
        err_msg = "Choose one of `as_hex` or `as_rgb`."
        assert (as_hex or as_rgb) is True and not (as_hex and as_rgb) is True, err_msg
        hex_value = {
            0: "#DDDDDD",  # some white color
            2: "#FAE7E0",
            4: "#F5E5CE",
            8: "#FEB17D",
            16: "#EB8E53",
            32: "#F87A63",
            64: "#E95839",
            128: "#F3D96B",
            256: "#F1D04B",
            512: "#E4C02A",
            1024: "#ECC400",
            2048: "#F46575",
            4096: "#F34B5C",
        }[value]

        if as_rgb:
            h = hex_value
            rgb_value = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
            return rgb_value
        else:
            return hex_value

    @staticmethod
    @lru_cache()
    def tile_representation(value: int, height: int, width: int):
        """Get a tile representation, which is an np.ndarray of shape `height,
        width` and np.uint8 dtype

        We use the lru_cache to avoid future computations, because we probably
        will only use this for values 2, 4, 8 ...
        """
        bg_color = Tile.get_color_code(value, as_hex=True)

        # PIL doesn't know how to search for installed fonts, so we actually need to
        # provide absolut path to on (AFAIK).
        font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
        font = ImageFont.truetype(font_path)

        text_size = font.getsize(str(value))
        th, tw = text_size
        if th > height or tw > width:
            raise ValueError("can't fit text in tile. Make bigger")

        im = Image.new("RGB", (height, width))
        draw = ImageDraw.Draw(im)

        draw.rectangle([height, width, 0, 0], fill=bg_color)
        text_loc = ((height - th) // 2, (width - tw) // 2)
        draw.text(text_loc, str(value))
        rgb_arr = np.array(list(im.getdata())).reshape(height, width, 3)
        return rgb_arr.astype(np.uint8)
