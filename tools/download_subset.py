"""Fetch a deterministic MUSDB18-HQ test subset using ZIP byte ranges.

Only audio from the official archive is downloaded. Every extracted member is
checked against the ZIP's uncompressed size and CRC32. No training data are used.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import struct
import subprocess
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor

URL = 'https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1'
ARCHIVE_BYTES = 22656664047


def fetch(start, end, output):
    chunk_size = 16 * 1024 * 1024
    if end-start+1 > chunk_size:
        chunks = [(lo, min(lo+chunk_size-1,end)) for lo in range(start,end+1,chunk_size)]
        def part(bounds):
            lo,hi=bounds
            path=output.parent/f'chunk_{lo}_{hi}.bin'
            if not path.exists() or path.stat().st_size != hi-lo+1:
                fetch(lo,hi,path)
            return path
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths=list(pool.map(part,chunks))
        with output.open('wb') as f:
            for path in paths:
                with path.open('rb') as source:
                    while block:=source.read(1024*1024):f.write(block)
                path.unlink()
                path.with_suffix('.headers').unlink(missing_ok=True)
        return
    headers = output.with_suffix('.headers')
    subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '3',
                    '--max-time', '600', '--range', f'{start}-{end}',
                    '-D', str(headers), URL, '-o', str(output)], check=True)
    header_text = headers.read_text().lower()
    if f'content-range: bytes {start}-{end}/{ARCHIVE_BYTES}' not in header_text:
        raise RuntimeError('Server did not honor the requested byte range')
    if output.stat().st_size != end - start + 1:
        raise RuntimeError('Incomplete range response')


def index(tail):
    """Decode ZIP64 central entries without downloading the whole archive."""
    pos = tail.find(b'PK\x01\x02')
    entries = []
    while tail[pos:pos + 4] == b'PK\x01\x02':
        h = struct.unpack_from('<4s6H3I5H2I', tail, pos)
        _, _, _, flags, method, _, _, crc, size, raw_size, nl, el, cl, _, _, _, offset = h
        name = tail[pos + 46:pos + 46 + nl].decode('utf-8')
        extra = tail[pos + 46 + nl:pos + 46 + nl + el]
        ep = 0
        while ep + 4 <= len(extra):
            tag, length = struct.unpack_from('<HH', extra, ep)
            payload = extra[ep + 4:ep + 4 + length]
            if tag == 1:
                zp = 0
                vals = [raw_size, size, offset]
                for k in range(3):
                    if vals[k] == 0xffffffff:
                        vals[k] = struct.unpack_from('<Q', payload, zp)[0]
                        zp += 8
                raw_size, size, offset = vals
            ep += 4 + length
        entries.append(dict(name=name, offset=offset, size=size,
                            raw_size=raw_size, crc=crc, method=method, flags=flags))
        pos += 46 + nl + el + cl
    if len(entries) != 902:
        raise RuntimeError(f'Unexpected archive member count: {len(entries)}')
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/MUSDB18-HQ'))
    p.add_argument('--tracks', type=int, default=5)
    p.add_argument('--index-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tail_file = tmp / 'tail.bin'
        fetch(ARCHIVE_BYTES - 200000, ARCHIVE_BYTES - 1, tail_file)
        entries = index(tail_file.read_bytes())
        tracks = sorted({PurePosixPath(e['name']).parts[1] for e in entries
                         if e['name'].startswith('test/') and e['name'].endswith('.wav')})
        selected = tracks[:args.tracks]
        manifest = dict(source=URL, archive_bytes=ARCHIVE_BYTES,
                        selection='First N test tracks in lexicographic order; no score-based selection',
                        tracks=selected, files=[])
        print(json.dumps(manifest, indent=2), flush=True)
        if args.index_only:
            return
        for track in selected:
            members = sorted([e for e in entries if e['name'].startswith(f'test/{track}/')
                              and e['name'].endswith('.wav')], key=lambda e: e['offset'])
            if len(members) != 5:
                raise RuntimeError(f'Incomplete track in archive: {track}')
            track_dir = args.output / 'test' / track
            if all((track_dir / Path(e['name']).name).exists() and
                   (track_dir / Path(e['name']).name).stat().st_size == e['raw_size']
                   for e in members):
                print(f'Already present: {track}', flush=True)
                manifest['files'].extend(members)
                continue
            # Last member's header and compressed payload; one contiguous request per track.
            start = members[0]['offset']
            last = members[-1]
            end = last['offset'] + 30 + len(last['name'].encode()) + 4096 + last['size'] - 1
            archive_part = tmp / 'track.bin'
            print(f'Downloading {track}: {(end-start+1)/1e6:.1f} MB', flush=True)
            fetch(start, end, archive_part)
            track_dir.mkdir(parents=True, exist_ok=True)
            with archive_part.open('rb') as f:
                for e in members:
                    f.seek(e['offset'] - start)
                    h = f.read(30)
                    if h[:4] != b'PK\x03\x04':
                        raise RuntimeError('Bad local ZIP header')
                    nl, el = struct.unpack_from('<HH', h, 26)
                    f.read(nl + el)
                    compressed = f.read(e['size'])
                    raw = zlib.decompress(compressed, -15) if e['method'] == 8 else compressed
                    if len(raw) != e['raw_size'] or zlib.crc32(raw) != e['crc']:
                        raise RuntimeError(f'CRC/length mismatch: {e["name"]}')
                    (track_dir / Path(e['name']).name).write_bytes(raw)
                    manifest['files'].append(e)
            print(f'CRC verified: {track}', flush=True)
            (args.output / 'subset_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
