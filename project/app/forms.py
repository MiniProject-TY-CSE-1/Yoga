from .models import Video
from django import forms
from .models import Image

class Video_form(forms.ModelForm):
   class Meta:
      model=Video
      fields=['video']
class Image_form(forms.ModelForm):
   class Meta:
      model=Image
      fields=['image']