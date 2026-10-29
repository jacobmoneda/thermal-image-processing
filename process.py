from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 256
HEIGHT = 192
PIXELS = WIDTH * HEIGHT


def extract_uti260e_temperatures(filename):
    data = Path(filename).read_bytes()

    # The UTi260E file contains two JPEGs.
    # Find the end of the final embedded JPEG.
    jpeg_end = data.rfind(b"\xff\xd9")

    if jpeg_end == -1:
        raise ValueError("Could not find JPEG end marker")

    payload = data[jpeg_end + 2:]

    # First 4 bytes after the JPEG are not part of
    # the temperature array.
    thermal_data = payload[4:4 + PIXELS * 8]

    if len(thermal_data) != PIXELS * 8:
        raise ValueError("Unexpected UTi260E thermal payload size")

    # Each pixel:
    # 4-byte big-endian float temperature
    # 4-byte auxiliary value
    dtype = np.dtype([
        ("temperature", ">f4"),
        ("aux", ">u4")
    ])

    records = np.frombuffer(
        thermal_data,
        dtype=dtype,
        count=PIXELS
    )

    temperature = records["temperature"].astype(
        np.float32
    )

    return temperature.reshape(HEIGHT, WIDTH)


temperature = extract_uti260e_temperatures(
    "00015E.jpg"
)

print("Shape:", temperature.shape)
print("Minimum:", temperature.min())
print("Maximum:", temperature.max())
print("Average:", temperature.mean())
print("Center:", temperature[96, 128])

np.savetxt(
    "00015E_temperature.csv",
    temperature,
    delimiter=",",
    fmt="%.4f"
)

plt.imshow(
    temperature,
    interpolation="nearest"
)

plt.colorbar(label="Temperature (°C)")
plt.title("UTi260E Temperature Map")
plt.xlabel("Thermal pixel X")
plt.ylabel("Thermal pixel Y")

plt.show()