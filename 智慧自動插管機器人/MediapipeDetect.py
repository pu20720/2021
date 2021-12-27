import cv2
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 算多邊形面積
def polygon_area(points):   
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2

##For static images:  對於靜態圖像
#IMAGE_FILES = []
#dirs = os.listdir("input")

#for d in dirs:
#    files = os.listdir("input/{d}".format(d=d))
#    for i in files:
#      img_path = os.path.join('input', d, i)
#      IMAGE_FILES.append(img_path)
        
#count=0
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#with mp_face_mesh.FaceMesh(
#    static_image_mode=True,  #如果設置為false，該解決方案會將輸入圖像視為視頻流。它將嘗試檢測第一張輸入圖像中的人臉，
#                             #並在成功檢測後進一步定位人臉地標。在隨後的圖像中，一旦檢測到所有max_num_faces 人臉並定位了相應的人臉地標，
#                             #它就會簡單地跟踪這些地標，而不會調用另一個檢測，直到它失去對任何人臉的跟踪。這減少了延遲，非常適合處理視頻幀。
#                             #如果設置為true，人臉檢測會在每個輸入圖像上運行，非常適合處理一批靜態的、可能不相關的圖像。默認為false。
#    max_num_faces=1,  #要檢測的最大人臉數。默認為1
#    refine_landmarks=True,   # MIN_DETECTION_CONFIDENCE：來自人臉檢測模型的最小置信值([0.0, 1.0])，以便將檢測視為成功。默認為0.5
#    min_detection_confidence=0.5) as face_mesh:
#  for idx, file in enumerate(IMAGE_FILES):
#    image = cv2.imread(file)
#    # Convert the BGR image to RGB before processing.
#    # 處理前將 BGR 圖像轉換為 RGB (因opencv讀取圖片的默認像素排列是BGR，和很多其他程式不一致，需要轉換)
#    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#    # Print and draw face mesh landmarks on the image.   印出Img和繪製面部網格地標 在圖像上。
#    if not results.multi_face_landmarks:
#      continue
#    annotated_image = image.copy()
#    for face_landmarks in results.multi_face_landmarks:
#      #print('face_landmarks:', face_landmarks)
#      # # 網格
#      # mp_drawing.draw_landmarks(
#      #     image=annotated_image,
#      #     landmark_list=face_landmarks,
#      #     connections=mp_face_mesh.FACEMESH_TESSELATION,
#      #     landmark_drawing_spec=None,
#      #     connection_drawing_spec=mp_drawing_styles
#      #     .get_default_face_mesh_tesselation_style())
#      #　輪廓
#      mp_drawing.draw_landmarks(
#          image=annotated_image,
#          landmark_list=face_landmarks,
#          connections=mp_face_mesh.FACEMESH_CONTOURS,
#          landmark_drawing_spec=None,
#          connection_drawing_spec=mp_drawing_styles
#          .get_default_face_mesh_contours_style())
#      # #瞳孔
#      # mp_drawing.draw_landmarks(
#      #     image=annotated_image,
#      #     landmark_list=face_landmarks,
#      #     connections=mp_face_mesh.FACEMESH_IRISES,
#      #     landmark_drawing_spec=None,
#      #     connection_drawing_spec=mp_drawing_styles
#      #     .get_default_face_mesh_iris_connections_style())

##     count += 1
##     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#    # save img
#    path = os.path.join('annotated_image', d)
#    if not os.path.exists(path):
#        os.mkdir(path)
#    out_path = os.path.join('annotated_image', d, i)
#    cv2.imwrite(out_path, annotated_image)
#    print(polygon_area(face_landmarks))

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
     max_num_faces=1,
     refine_landmarks=True,
     min_detection_confidence=0.5,    # MIN_TRACKING_CONFIDENCE：來自地標跟踪模型的最小置信值([0.0, 1.0])，用於將被視為成功跟踪的人臉地標，
                                      # 否則將在下一個輸入圖像上自動調用人臉檢測。將其設置為更高的值可以提高解決方案的穩健性，但代價是更高的延遲。
                                      # 如果static_image_mode 為true，則忽略，人臉檢測在每個圖像上運行。默認為0.5。
     min_tracking_confidence=0.5) as face_mesh:
   while cap.isOpened():
     success, image = cap.read()
     if not success:
       print("Ignoring empty camera frame.")
       # If loading a video, use 'break' instead of 'continue'.
       continue

     # To improve performance, optionally mark the image as not writeable to
     # pass by reference.
     image.flags.writeable = False
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     results = face_mesh.process(image)

     # Draw the face mesh annotations on the image.
     image.flags.writeable = True
     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     if results.multi_face_landmarks:
       for face_landmarks in results.multi_face_landmarks:
         mp_drawing.draw_landmarks(
             image=image,
             landmark_list=face_landmarks,
             connections=mp_face_mesh.FACEMESH_TESSELATION,
             landmark_drawing_spec=None,
             connection_drawing_spec=mp_drawing_styles
             .get_default_face_mesh_tesselation_style())
         mp_drawing.draw_landmarks(
             image=image,
             landmark_list=face_landmarks,
             connections=mp_face_mesh.FACEMESH_CONTOURS,
             landmark_drawing_spec=None,
             connection_drawing_spec=mp_drawing_styles
             .get_default_face_mesh_contours_style())
         mp_drawing.draw_landmarks(
             image=image,
             landmark_list=face_landmarks,
             connections=mp_face_mesh.FACEMESH_IRISES,
             landmark_drawing_spec=None,
             connection_drawing_spec=mp_drawing_styles
             .get_default_face_mesh_iris_connections_style())
     # Flip the image horizontally for a selfie-view display.
     cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
     if cv2.waitKey(5) & 0xFF == 27:
         break

cap.release()