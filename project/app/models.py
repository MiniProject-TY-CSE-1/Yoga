from django.db import models

# Create your models here.
class Video(models.Model):
   video=models.FileField(upload_to="video/")
   def __str__(self):
      return self.caption
   
class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    def __str__(self):
        return self.title