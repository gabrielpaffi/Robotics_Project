import cv2
from ultralytics import YOLO

model = YOLO("robotv12.pt")
class_object = ["blackholes","earth","mars", "robot" ]
order = {'robot': 0, 'mars': 1, 'earth': 2, 'blackholes': 3}

circle_radius = 3
circle_thicknes = 3
line_thickness = 2
blue= (255,0,0)
maps_size_width = 100
frame_width = 640
frame_height = 480
scaling_factor = frame_width/maps_size_width

frames_per_second = 10


def detection(frame):

    confidence_min = 0.55

    results = model.track(frame, conf = confidence_min, verbose = False)

    annotated_frame = results[0].plot()


    array_box_dim_tot = []
           
   

    for objects in results:
    
            number_objects = len(objects.boxes.cls)
            box_dimensions = objects.boxes.xywhn
            for i in range(number_objects) :
              
                array_box_dim_object = []
                confidence = round(objects.boxes.conf[i].item(),3)
                cls = objects.boxes.cls[i].item()
                array_box_dim = box_dimensions[i].tolist() # center x,y width, height
                for i in range(len(array_box_dim)):
                    array_box_dim[i] = round(array_box_dim[i],2)
                array_box_dim_object = [class_object[int(cls)], array_box_dim, confidence]

                array_box_dim_tot.append(array_box_dim_object)


                        

    return(array_box_dim_tot, annotated_frame)



def annotation(frame, coord):
    

    for k in range(len(coord)-1):
        
        
        frame = cv2.circle(frame, coord[k], circle_radius, blue,circle_thicknes)
        frame = cv2.line(frame, coord[k], coord[k+1], blue, line_thickness)  
        
        

    
    return frame


def transformation(coords):
    scaled_coords = []
    for coord in coords:

        scaled_x =  int(scaling_factor * coord[0])
        scaled_y = frame_height - int(scaling_factor * coord[1])
        scaled_coord = (scaled_x, scaled_y)
        scaled_coords.append(scaled_coord)


    return scaled_coords




