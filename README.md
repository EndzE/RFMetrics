# RFMetrics

**Desktop video quality metrics.** Compare distorted files against a reference using industry-standard algorithms, complete with per-frame plots and a worst-frames viewer.

**Supported Metrics:**  
`PSNR` • `SSIM` • `XPSNR` • `VMAF` • `SSIMULACRA2` • `Butteraugli` • `CVVDP`

---

## Requirements

- **`ffmpeg` + `ffprobe`**
- **`FFVship` binary**: Required for **SSIMULACRA2**, **Butteraugli**, and **CVVDP** *(optional if you do not need these specific metrics)*.

---

## How to Run

1. **Download** the latest archive from the [Releases](#) page.
2. **Extract** the archive into a folder of your choice.
3. **Install FFmpeg**: 
   - Download binaries from [Gyan.dev](https://github.com/GyanD/codexffmpeg/releases) or [BtbN Builds](https://github.com/BtbN/FFmpeg-Builds/releases).
   - Place the `ffmpeg` and `ffprobe` binaries into a `ffmpeg/` folder next to the `.exe`, or add them to your system's `%PATH%`.
4. **Install FFVship** *(Optional)*:
   - Download binaries matching your GPU from [Codeberg](https://codeberg.org/Line-fr/Vship/releases).
   - Place the `ffvship` binaries into a `ffvship/` folder next to the `.exe`, or add them to your system's `%PATH%`.
5. **Launch** `rfmetrics.exe`.
6. **Analyze**:
   - Queue your distorted files *(Drag & drop supported)*.
   - Set your **reference** video.
   - Tick the headers for the metrics you want to calculate.
   - Click **Start**.

---

## Notes

- **Worst-Frames Extraction**: Extracts the worst-N frames as PNGs into a temporary per-process directory (auto-deleted when the viewer is closed). Use the **Export** buttons to save permanent copies to a custom folder or directly beside each video file.
- **VMAF Models**: VMAF JSON models are read from the `vmaf-models/` directory next to the executable. A built-in fallback model is included by default.
- **Visibility Controls**: 
  - Toggling graphs off in the **Legend** simply hides them from the plot view.
  - Unchecking the box next to a **file** hides it from the plot view, as well as from the *Save PNG* and *Copy* functions.

---

## FAQ

> **Is it vibecoded?**
> 
> *Yes.*

---

## Original Inspiration

[**FFMetrics**](https://github.com/fifonik/FFMetrics).