import cv2
import mediapipe as mp
import numpy as np
from django.shortcuts import render,HttpResponse
from .forms import Video_form
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#from detections
def TPose(sample_img) :
    tpose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
    #T-Pose image read
    image_height, image_width, _ = sample_img.shape

    # Perform pose detection after converting the image into RGB format.
    results = tpose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    landmarks=results.pose_landmarks.landmark

    return landmarks,image_height,image_width

def isMatched(landmarks,h,w,TPoselandmarks,height,width) :
    for i in range(33):
        # print(landmarks[i].x*w,TPoselandmarks[i].x*width)
        a=abs(landmarks[i].x*w-TPoselandmarks[i].x*width)
        b=abs(landmarks[i].y*h-TPoselandmarks[i].y*height)
        c=abs(landmarks[i].visibility-TPoselandmarks[i].visibility)
        if a>20.0 or b>20.0 or c>5.0:
            return False
    return True

def handleuploadedfile(video,sample_img) :
    cap = cv2.VideoCapture(video)
    TPoselandmarks,height,width = TPose(sample_img)
     # Check if the input video opened successfully
    if not cap.isOpened():
        print("Error: Could not open input video.")
        exit()
     
     # Get video properties (frame width, height, and frames per second)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
     


     # Define the codec and create VideoWriter for the output video
    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

     # Check if the VideoWriter opened successfully
    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        cap.release()
        exit()

    # Loop through frames, process (if needed), and write to the output video
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        # Process the frame (e.g., apply filters or modifications) if needed
            #recolor image to RGB (opencv gives frame in BGR but mediapipe wants in RGB)
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        image.flags.writeable=False


        #make detections    
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(image)
        #recolor image to BGR
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks=results.pose_landmarks.landmark
            h,w,_=image.shape
            ans=isMatched(landmarks,100,100,TPoselandmarks,100,100)
            if ans==True:
                cv2.putText(image,'Yes',
                            tuple(np.multiply([1,1],[100,100]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                # break
            else :
                cv2.putText(image,'No',
                            tuple(np.multiply([1,1],[100,100]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4,cv2.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                         mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                         mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                        )      

        out.write(image)

    cap.release()
    out.release()
    return output_video_path


from django.http import JsonResponse

# Create your views here.
def index(request):
    print("Request")

    # all_videos = Video.objects.all()
    # if request.method=="POST":
    #     form = Video_form(request.POST,request.FILES)
    #     if form.is_valid():
    #         video = form.save()
    #         processing_video=video.video.path
    #         return HttpResponse("<h1>Uploaded Successfully</h1>")
    # else:
    #     form = Video_form()
    # return render(request,'index.html',{'form':form,"all":all_videos})
    return JsonResponse({"Request received"})