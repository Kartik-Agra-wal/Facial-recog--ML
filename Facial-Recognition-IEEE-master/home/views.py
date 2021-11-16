from django.views.decorators import gzip
from django.shortcuts import render
from home.camera import VideoCamera
from django.http.response import StreamingHttpResponse

def Index(request):
	return render(request, 'home/index.html')

@gzip.gzip_page
def webcam_feed(request):
	try:
		cam = VideoCamera()
		return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
	except:
		pass


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


