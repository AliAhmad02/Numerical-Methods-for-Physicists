"""Final project: Fourier optics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.lib.index_tricks import IndexExpression
from numpy.typing import NDArray
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy.fft import fft2
from scipy.fft import fftfreq
from scipy.fft import ifft2


def gaussian(x, mu=0, sig=2):
    """Gaussian function scaled such that the values are between 0 and 1."""
    G: NDArray[np.float64] = np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    return G / np.amax(G)


def get_Efield(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    A0: NDArray[np.float64],
    dist: float,
    wavelength: float,
) -> NDArray[np.float64]:
    """Get the electric field based on the slit and light configuration."""
    # Generate aperature matrix
    A: NDArray[np.float64] = fft2(A0)
    # Calculate wavenumber from wavelength
    k: NDArray[np.float64] = 2 * np.pi / wavelength
    # Calculate the fourier frequencies
    fourier_freqx: NDArray[np.float64] = fftfreq(len(x), np.diff(x)[0])
    fourier_freqy: NDArray[np.float64] = fftfreq(len(y), np.diff(y)[0])
    # Convert fourier frequencies to angular frequencies
    kx: NDArray[np.float64] = 2 * np.pi * fourier_freqx
    ky: NDArray[np.float64] = 2 * np.pi * fourier_freqy
    # Create 2D meshgrids for fourier frequencies
    kxs, kys = np.meshgrid(kx, ky)
    # Take inverse fourier transform to calculate the field
    E = ifft2(
        A
        * np.exp(
            1j * dist * np.sqrt(k**2 - kxs**2 - kys**2),
        ),
    )
    return np.abs(E)


def create_word_image(
    word: str,
    size: tuple[int, int],
    scaling: float = 1,
    font_name: str = "DejaVuSans-Bold.ttf",
) -> NDArray[np.float64]:
    """Create a 2D numpy array representing a word/sentence.

    The value of this array will be zero everywhere except for
    the positions corresponding to the pixels that make up the
    letters in our word/sentence where the value will be 1.
    """
    # Create a blank black image
    img: Image = Image.new("1", size, color=0)
    draw: ImageDraw = ImageDraw.Draw(img)

    # Calculate the font size relative to the image size
    font_size: int = scaling * min(size) // len(word)

    font: ImageFont = ImageFont.truetype(font_name, font_size)

    # Calculate the size of the word with the adjusted font size
    ascent, descent = font.getmetrics()
    text_width: float = font.font.getsize(word)[0][0]
    text_height: float = ascent + descent

    # Calculate the position to center the word horizontally
    position_x: int = (size[0] - text_width) // 2

    # Calculate the position to center the word vertically
    position_y: int = (size[1] - text_height) // 2

    # Draw the white word on the background
    draw.text((position_x, position_y), word, fill=1, font=font)

    # Flipping the image
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return np.array(img)


def calculate_and_plotE(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    A0: NDArray[np.float64],
    dist: float,
    wavelength: float,
    slice_x: IndexExpression | None = None,
    slice_y: IndexExpression | None = None,
) -> None:
    """Calculate and plot diffraction pattern and intensity distribution."""
    if not slice_x and not slice_y:
        raise Exception("Must include either an x- or y-slice!")
    xs, ys = np.meshgrid(x, y)
    E: NDArray[np.float64] = get_Efield(x, y, A0, dist, wavelength)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.axis("off")
    ax1.imshow(E, cmap="hot")
    if slice_x:
        ax2.plot(xs[slice_x], E[slice_x] ** 2)
    elif slice_y:
        ax2.plot(ys[slice_y], E[slice_y] ** 2)
    plt.show()


def animate_with_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    A0: NDArray[np.float64],
    distances: NDArray[np.float64],
    wavelength: float,
) -> None:
    """Calculate and animate diffraction pattern with distance."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    def animate(i):
        """Animation function."""
        E = get_Efield(x, y, A0, distances[i], wavelength)
        ax.imshow(E, cmap="hot")
        ax.set_title(f"Distance={distances[i]:.2f}")

    ani = FuncAnimation(
        fig,
        animate,
        frames=len(distances),
        interval=20,
    )
    ani.save("basic_animation.GIF", fps=27, writer="pillow", dpi=300)


def do_multiple_animations(
    Apertures: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    distances: NDArray[np.float64],
    wavelength: float,
    rows: int,
    cols: int,
) -> None:
    """Perform animations for multiple apertures."""
    figsize: tuple[int, int] = (rows * 4, cols * 4)
    n_plots: int = len(Apertures)
    idx_list: int = list(range(1, n_plots + 1))
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(True)

    def animate(i):
        print(i)
        for j in range(n_plots):
            ax = fig.add_subplot(rows, cols, idx_list[j])
            E = get_Efield(x, y, Apertures[j], distances[i], wavelength)
            ax.imshow(E, cmap="hot")
            ax.axis("off")
        fig.suptitle(f"Distance={distances[i]:.2f}", fontsize=35)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ani = FuncAnimation(
        fig,
        animate,
        frames=len(distances),
        interval=20,
    )
    ani.save("basic_animation.GIF", fps=30, writer="pillow", dpi=300)


# We write everything in units of 10⁻4 meters
wavelength: int = 700e-5
dist: int = 300

start: int = -20
end: int = 20
N: int = 1500

x: NDArray[np.float64] = np.linspace(start, end, N)
y: NDArray[np.float64] = np.linspace(start, end, N)

xs, ys = np.meshgrid(x, y)

slice_x = np.s_[1000, :]
slice_y = np.s_[:, 1000]

slit: str = "┃"
double_slit: str = "┃┃"
grating: str = "▩"
doubleO: str = "OO"
hexagon: str = "⎔"
amogus: str = "ඞ"
david: str = "✡"
rub: str = "۞"
optics: str = "OPTICS SUCKS"
handicap: str = "♿"
gauss_vals: NDArray[np.float64] = gaussian(y)

A0_slit: NDArray[np.float64] = create_word_image(slit, (N, N), scaling=0.4)
A0_doubleslit: NDArray[np.float64] = create_word_image(
    double_slit,
    (N, N),
    scaling=0.2,
)

A0_grating: NDArray[np.float64] = create_word_image(
    grating,
    (N, N),
    scaling=0.5,
)

A0_gauss: NDArray[np.float64] = np.array(
    [A0_slit[i] * gauss_vals[i] for i in range(N)],
)
A0_doubleO: NDArray[np.float64] = create_word_image(
    doubleO,
    (N, N),
    scaling=0.3,
)
A0_hexagon: NDArray[np.float64] = create_word_image(
    hexagon,
    (N, N),
    scaling=0.6,
)
A0_amogus: NDArray[np.float64] = create_word_image(
    amogus,
    (N, N),
    font_name="lklug.ttf",
    scaling=0.8,
)
A0_david: NDArray[np.float64] = create_word_image(david, (N, N), scaling=0.8)
A0_rub_har: NDArray[np.float64] = create_word_image(
    rub,
    (N, N),
    font_name="Harmattan-Bold.ttf",
    scaling=0.5,
)
A0_rub_nak: NDArray[np.float64] = create_word_image(
    rub,
    (N, N),
    font_name="NotoNaskhArabicUI-SemiBold.ttf",
    scaling=0.6,
)
A0_optics: NDArray[np.float64] = create_word_image(optics, (N, N))

A0_handicap: NDArray[np.float64] = create_word_image(
    handicap,
    (N, N),
    scaling=0.7,
)

distances = np.linspace(1, 1000, 3)
Apertures = [
    A0_grating,
    A0_doubleO,
    A0_hexagon,
    A0_amogus,
    A0_david,
    A0_rub_har,
    A0_rub_nak,
    A0_optics,
    A0_handicap,
]
do_multiple_animations(Apertures, x, y, distances, wavelength, 3, 3)

calculate_and_plotE(x, y, A0_slit, dist, wavelength, slice_x)
calculate_and_plotE(x, y, A0_gauss, dist, wavelength, slice_y=slice_y)
calculate_and_plotE(x, y, A0_grating, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_amogus, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_doubleslit, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_doubleO, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_hexagon, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_david, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_rub_har, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_rub_nak, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_optics, dist, wavelength, slice_x=slice_x)
calculate_and_plotE(x, y, A0_handicap, dist, wavelength, slice_x=slice_x)
