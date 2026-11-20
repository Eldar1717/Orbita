"""Orbita: static UI + local Russian Piper speech. python server.py"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os, io, json, threading, wave, urllib.request, functools
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parent
CACHE=Path.home()/'.cache'/'orbita'; CACHE.mkdir(parents=True,exist_ok=True)
lock=threading.Lock()
@functools.lru_cache(maxsize=1)
def voice(name):
    from piper import PiperVoice
    from piper.config import PiperConfig
    import onnxruntime as ort
    base=f'ru_RU-{name}-medium.onnx'; path=CACHE/base
    for suffix in ('.json',''):
        dest=CACHE/(base+suffix)
        if not dest.exists():
            url=f'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/{name}/medium/{dest.name}?download=true'
            with urllib.request.urlopen(url,timeout=90) as src, open(str(dest)+'.part','wb') as out:
                import shutil; shutil.copyfileobj(src,out)
            os.replace(str(dest)+'.part',dest)
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
    return PiperVoice(session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']),config=PiperConfig.from_dict(json.loads(Path(str(path)+'.json').read_text())),download_dir=CACHE)
class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path!='/api/speech':return self.send_error(404)
        try:
            n=int(self.headers.get('Content-Length',0)); assert 0<n<=16000
            j=json.loads(self.rfile.read(n)); name=j['voice']; t=j['text']; rate=float(j['rate'])
            assert name in ('irina','dmitri','ruslan','denis') and isinstance(t,str) and 0<len(t)<=1000 and .7<=rate<=1.4
        except Exception:return self.send_error(400)
        try:
            from piper.config import SynthesisConfig
            buf=io.BytesIO()
            with lock,wave.open(buf,'wb') as wav:voice(name).synthesize_wav(t,wav,SynthesisConfig(length_scale=1/rate))
            data=buf.getvalue();self.send_response(200);self.send_header('Content-Type','audio/wav');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        except (ConnectionResetError,BrokenPipeError): pass
        except Exception as e: print('Piper:',e,flush=True);self.send_error(503)
if __name__=='__main__':
    os.chdir(ROOT);host=os.getenv('ORBITA_HOST','127.0.0.1');port=int(os.getenv('ORBITA_PORT',8000));print(f'Orbita: http://{host}:{port}',flush=True);ThreadingHTTPServer((host,port),Handler).serve_forever()
