# Initial audit and continuation record — 8 November 2026

Original project: `/Users/jiajunliu/Desktop/5305/project`.
Continuation: `/Users/jiajunliu/Documents/ChatGPT/5305/project`.
Original tracked state was commit `500132c`; no tracked modifications were present.
The untracked progress-report PDF was preserved in the original directory.

The workspace root's Report 2 is a different assessment and was already packaged.
The project concerns acoustic limitations of vocal source separation. A progress
report had revised the initial "masking versus deep learning" comparison to a
question about time-frequency overlap and mixture-phase errors, with Open-Unmix
and HTDemucs as pretrained reference systems.

At inspection, the original project had eight small Python modules, an installed
Python 3.9 environment, no generated results, no installed pretrained systems,
and an older README that still described the original research direction.

The local `data/musdb18hq.zip` is 5,304,030,672 bytes. It begins as a ZIP but has
no end-of-central-directory record: the download is incomplete. Complete local
headers cover training tracks; the next vocal member is truncated. The official
archive has 22,656,664,047 bytes. The original partial download was left unchanged.

A new Python 3.12 environment was created in the continuation directory. A
range downloader fetches the lexicographically first five official test tracks
without downloading the complete archive. Every member is verified using the
central-directory uncompressed length and CRC32. The deterministic selection is
recorded in `data/MUSDB18-HQ/subset_manifest.json`.

The loader now reads only requested samples, validates alignment and interval
boundaries, and retains mixture-versus-stem-sum discrepancy measurements. All
experimental mixtures are reconstructed from the same stems for exact additivity.
All methods process identical input samples. Analysis and model-native STFTs are
reported separately. Reconstruction SDR and SI-SDR have explicit definitions;
BSS Eval v4 is evaluated separately rather than conflated with either metric.
