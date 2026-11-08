"""Build a captioned demonstration with real execution logs and shared-gain audio."""
from pathlib import Path
import json
import subprocess
import textwrap
import numpy as np
import soundfile as sf
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/video';OUT.mkdir(parents=True,exist_ok=True)
TMP=OUT/'frames';TMP.mkdir(exist_ok=True)
R=ROOT/'results/study'
S=json.loads((R/'summary.json').read_text())
CX=json.loads((R/'context_summary.json').read_text())
EVENTS=json.loads((ROOT/'results/demo/execution_events.json').read_text())
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
MONO='/System/Library/Fonts/Menlo.ttc'
import matplotlib
_fonts=Path(matplotlib.get_data_path()) / "fonts/ttf"
if not Path(FONT).exists():FONT=str(_fonts/"DejaVuSans.ttf")
if not Path(BOLD).exists():BOLD=str(_fonts/"DejaVuSans-Bold.ttf")
if not Path(MONO).exists():MONO=str(_fonts/"DejaVuSansMono.ttf")
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
chapters=[]
def font(n,bold=False,mono=False):return ImageFont.truetype(MONO if mono else BOLD if bold else FONT,n)
def base(title,subtitle='ELEC5305 | Jiajun Liu | 550378323'):
    im=Image.new('RGB',(1280,720),'#f5f8fb');d=ImageDraw.Draw(im)
    d.rectangle([0,0,1280,130],fill='#153b55');d.text((56,28),title,font=font(36,True),fill='white')
    d.text((58,85),subtitle,font=font(21),fill='#d3e7f4')
    d.line((55,665,1225,665),fill='#becedb',width=2)
    d.text((58,683),'Vocal source separation | measured results, explicit limitations',font=font(17),fill='#527087')
    return im,d

def paragraph(d,text,x,y,width=1100,size=29,color='#233c50',gap=13):
    words=text.split();lines=[];line=''
    for word in words:
        test=(line+' '+word).strip()
        if d.textlength(test,font=font(size))>width:lines.append(line);line=word
        else:line=test
    if line:lines.append(line)
    for line in lines:d.text((x,y),line,font=font(size),fill=color);y+=size+gap
    return y

def still(title,text,duration=12,graphic=None,subtitle=None):
    im,d=base(title,subtitle or 'ELEC5305 | Jiajun Liu | 550378323')
    if graphic:
        g=Image.open(graphic).convert('RGB');g.thumbnail((1130,420))
        im.paste(g,((1280-g.width)//2,150))
        paragraph(d,text,60,590,size=23,width=1160,gap=7)
    else:
        y=185
        for block in text.split('\n\n'):
            y=paragraph(d,block,60,y,size=29,width=1150)+22
    path=TMP/f'frame_{len(chapters):03d}.png';im.save(path)
    chapters.append(dict(frame=path,duration=duration,audio=None,title=title,text=text))

still('Why does vocal separation fail?',
      'One question: how do time-frequency overlap and mixture-phase mismatch relate to vocal reconstruction errors?\n\nWe reproduce oracle baselines, make a small phase-sensitive modification, and compare UMXHQ with HTDemucs.\n\nCaptioned demonstration with real local execution and music examples.',14)
still('A controlled, reproducible experiment',
      f'{S["n_tracks"]} official test tracks; {S["n_excerpts"]} active excerpts; {S["n_cases"]} paired inputs. Same stems and exact inputs for every method.',14,graphic=R/'figures/pipeline.png')
im,d=base('The small modification: phase-sensitive masking')
code=(ROOT/'src/oracle_methods.py').read_text()
start=code.index('def phase_sensitive_mask(');end=code.index('\ndef oracle_estimates',start)
# Select implementation lines rather than long docstrings.
lines=[line for line in code[start:end].splitlines() if not line.strip().startswith(('"""','The coefficient','the ordinary','waveform-'))]
lines=['def phase_sensitive_mask(X, V, upper=1.0):','    power = abs(X)**2',
       '    numerator = real(V * conj(X))','    m = divide(numerator, power, where=power > 0)',
       '    return clip(m, 0, upper)', '', '# Coefficient-wise projection onto the mixture direction.',
       '# upper=1: attenuation only; upper=inf: amplification allowed.']
d.rounded_rectangle([55,160,1225,475],radius=15,fill='#e5edf4')
for i,line in enumerate(lines):d.text((80,185+i*33),line,font=font(23,mono=True),fill='#163b54')
paragraph(d,'Readable pseudocode for src/oracle_methods.py. These oracle masks use true stems; they are diagnostic references, not blind separators.',60,515,size=26)
path=TMP/f'frame_{len(chapters):03d}.png';im.save(path)
chapters.append(dict(frame=path,duration=17,audio=None,title='Code: phase-sensitive projection',text='Oracle information access is explicit.'))
# Actual timestamped execution; short pauses are extended for legibility.
for i,event in enumerate(EVENTS):
    im,d=base('Code in action: recorded local execution','Actual stdout from python -m src.demo | brief pauses extended for readability')
    d.rounded_rectangle([45,150,1235,625],radius=14,fill='#132636')
    visible=EVENTS[max(0,i-17):i+1]
    for j,e in enumerate(visible):
        line=e['message']
        d.text((65,168+j*24),line,font=font(17,mono=True),fill='#b8edcf' if j==len(visible)-1 else '#d6e2eb')
    d.text((1030,95),f'run time {event["seconds"]:.1f} s',font=font(16),fill='#d3e7f4')
    path=TMP/f'frame_{len(chapters):03d}.png';im.save(path)
    dur=max(.8,EVENTS[i+1]['seconds']-event['seconds']) if i+1<len(EVENTS) else 6
    chapters.append(dict(frame=path,duration=dur,audio=None,title='Actual execution',text=event['message']))

# Fixed first track, fixed first excerpt: no listening example cherry-picking.
import pandas as pd
D=pd.read_csv(R/'metrics.csv')
first=D[(D.track=='AM Contra - Heart Peripheral')&(D.start_sec==30)&(D.condition=='natural')]
scores={x.method:x.reconstruction_sdr_db for x in first.itertuples()}
for name,label in [('mixture','Original mixture'),('reference','True vocal reference'),('irm','Power IRM (oracle)'),('magnitude_phase','True magnitude + mixture phase (oracle)'),('clipped_psm','Clipped PSM (oracle)'),('openunmix','Open-Unmix UMXHQ'),('htdemucs','HTDemucs')]:
    value='Reference stem' if name=='reference' else f'Reconstruction SDR: {scores[name]:.2f} dB'
    still('Listen: '+label,
          'AM Contra - Heart Peripheral | 30-36 seconds\n\n'+value+'\n\nIdentical playback gain across all examples. Use headphones.\n\nSource: MUSDB18-HQ, educational research example.',6)
    chapters[-1]['audio']=ROOT/'results/demo/audio'/f'{name}.wav'
still('Measured findings',
      f'Clipped PSM improves over power IRM by {S["clipped_psm_gain_mean_db"]:.2f} dB mean reconstruction SDR. Dots show track variability.',18,graphic=R/'figures/natural_scores.png')
still('A practical iteration: add context',
      f'Give both models 18 seconds; score the same centre six seconds. The weak HTDemucs case improves from {CX["failed_case"]["short_reconstruction_sdr_db"]:.2f} to {CX["failed_case"]["reconstruction_sdr_db"]:.2f} dB. Not every excerpt improves.',16,graphic=R/'figures/context_iteration.png')
for name,label in [('short_input','HTDemucs: original short input'),('with_context','HTDemucs: with surrounding context')]:
    still('Listen: '+label, 'AM Contra - Heart Peripheral | 60-66 seconds\n\nSame scored excerpt and shared playback gain.\n\nThe surrounding context is used at inference, without true stems.',6)
    chapters[-1]['audio']=R/'context_audio/comparison'/f'{name}.wav'
still('What can we conclude?',
      'Exact target magnitude is not automatically the best reconstruction when phase is wrong.\n\nGain changes move overlap and phase mismatch together; their independent causes are not isolated.\n\nFive tracks, short excerpts, unequal model training data and no formal listening study limit generalization.',20)
still('Reproduce and inspect',
      'python -m src.run_experiment\npython -m src.summarize\npython -m src.demo\n\nSource, per-case CSVs, manifests, report and sound examples accompany the project.\n\nPrior algorithms and pretrained models are credited in the report.',14)

# Encode clips with a uniform stream layout, then losslessly concatenate.
for i,chapter in enumerate(chapters):
    dur=float(chapter['duration'])
    audio_path=TMP/f'audio_{i:03d}.wav'
    if chapter['audio']:
        x,sr=sf.read(chapter['audio'],always_2d=True)
        fade=min(round(.01*sr),len(x)//2)
        x[:fade]*=np.linspace(0,1,fade)[:,None];x[-fade:]*=np.linspace(1,0,fade)[:,None]
    else:
        sr=44100;x=np.zeros((round(dur*sr),2))
    sf.write(audio_path,x,sr,subtype='PCM_16')
    clip=TMP/f'clip_{i:03d}.mp4'
    cmd=[ffmpeg,'-y','-loglevel','error','-loop','1','-framerate','12','-i',str(chapter['frame']),
         '-i',str(audio_path),'-t',f'{dur:.3f}','-c:v','libx264','-preset','fast','-crf','21',
         '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','44100','-ac','2','-r','12',str(clip)]
    subprocess.run(cmd,check=True)
    print(f'Encoded {i+1}/{len(chapters)}',flush=True)
concat=TMP/'concat.txt'
concat.write_text('\n'.join("file '"+str(TMP/f'clip_{i:03d}.mp4')+"'" for i in range(len(chapters))))
final=OUT/'ELEC5305_Project_Demo.mp4'
subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),
                '-c','copy','-movflags','+faststart',str(final)],check=True)
time=0;listing=[]
for c in chapters:
    listing.append(dict(start_seconds=time,duration_seconds=c['duration'],title=c['title'],caption=c['text']))
    time+=c['duration']
(OUT/'video_chapters.json').write_text(json.dumps(listing,indent=2))
(OUT/'README.md').write_text('Captioned project demonstration. No spoken narration is claimed. Terminal text is captured from an actual local execution; some pauses are lengthened for readability. The same fixed six-second excerpt and common gain are used for sound comparisons. Captions explain oracle information access and model/data limitations. Source music: MUSDB18-HQ, AM Contra - Heart Peripheral; see report for attribution.\n')
print(final,'duration',round(time,1),'s')
