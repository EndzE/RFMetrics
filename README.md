# RFMetrics

Desktop video quality metrics: compare distorted files against a reference (PSNR, SSIM, XPSNR, VMAF, SSIMULACRA2, Butteraugli, CVVDP), with per-frame plots and a worst-frames viewer.

## Requirements

- `ffmpeg` + `ffprobe` (in folder "ffmpeg" next to exe or in `PATH`)
- `FFVship` binary for SSIMULACRA2 / Butteraugli / CVVDP (optional)

## Run

1. Download archive from Releases
2. Extract it into folder
3. Download ffmpeg binaries (https://github.com/GyanD/codexffmpeg/releases or https://github.com/BtbN/FFmpeg-Builds/releases)
4. Put ffmpeg binaries into the folder "ffmpeg" next to exe or make them available in %PATH%
5. Download ffvship binaries based on your GPU https://codeberg.org/Line-fr/Vship/releases
6. Put ffvship binaries into the folder "ffvship" next to exe or make them available in %PATH%
7. Run rfmetrics.exe
8. Queue files (drag & drop works), set a reference, tick metric headers, Start.

## Notes

- Bad-frames Extract renders worst-N PNGs to a per-process tmp dir (auto-deleted on viewer close); Export buttons save copies to the chosen folder (or beside each file).
- VMAF JSON models are read from `vmaf-models/` next to the exe (built-in fallback included).
- Disabling graphs in Legend is just hiding them from the plot view, disabling by checkbox next to the file is hiding from view and from Save PNG and Copy functions

## FAQ

- Is it vibecoded?
- Yes

## Original inspiration

FFMetrics - https://github.com/fifonik/FFMetrics