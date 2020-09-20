from django.db import models

# Create your models here.

class Image(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to='images',blank=True, null=True)

    def publish(self):
        return self.id