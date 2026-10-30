"""Controller-only screenshot history; never changes actor observations/history."""
from bootstrap import ROOT
import hashlib,time
from pathlib import Path
from environments import BrowserCodeWorkspace,verify_png
class JudgedBrowserCodeWorkspace(BrowserCodeWorkspace):
 def enable_judge_capture(self,directory):
  self.judge_capture_directory=Path(directory)/'judge-screenshots';self.judge_capture_directory.mkdir(exist_ok=True);self.judge_images=[];self.judge_capture_seconds=0.;self.judge_capture_operations=0
 def execute(self,action,cwd=''):
  output=super().execute(action,cwd)
  if hasattr(self,'judge_capture_directory'):
   start=time.monotonic();image=self.judge_capture_directory/f'step-{len(self.judge_images):04d}.png'
   try:
    self.judge_capture_operations+=1;self.capture(image);verify_png(image);self.judge_images.append(image)
   finally:self.judge_capture_seconds+=time.monotonic()-start
  return output
