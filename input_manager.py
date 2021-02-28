import pygame
import numpy as np
import math
from CNN_model import Load_model_for_pygame, create_and_train_model
import os.path




run_code = True

#============================================
#Importing the ML model
#============================================

if os.path.isdir('number_recognition_CNN'):
    model = Load_model_for_pygame('number_recognition_CNN')
else:
    print("No model currently detected")
    yn = input("Enter 'y' if you want to create and train the CNN? (may take some time):")
    if(yn == 'y'):
        create_and_train_model()
        model = Load_model_for_pygame('number_recognition_CNN')
    else:
        run_code = False;






if run_code:

    print("==================================================================")
    print("- Draw a number in the black square using the left click")
    print("- Erase your drawing with left click")
    print("- Clear with mouse wheel click or the space bar")
    print("- Change brush size with mouse wheel")
    print("Try to draw a the center of the square")
    print("==================================================================")

    #============================================
    #Pygame UI
    #============================================

    pygame.init()
    pygame.font.init()
    
    #Screen settings
    
    #Define a grid of resolution 28x28
    grid_pixel_number = 28
    #Define the conversion ratio from the grid to real screen
    #That's what you want to change to change screen size
    grid_pixel_width = 15
    #Size of the square where you can write a number in real pixel
    drawing_size = grid_pixel_number*grid_pixel_width
    
    #Define histogram size
    histogram_size = drawing_size
    bar_width = histogram_size/10
    bar_height = 9*(histogram_size//10) #This is actually the maximum bar height representing 100%
    
    #Define the screen size
    screen_size = (drawing_size+histogram_size, drawing_size)    
    #Create the screen
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Digit Recognizer")
    
    #Defines the grid of 28x28
    grid = np.zeros((grid_pixel_number, grid_pixel_number))
    grid_display_mouse_pos = np.zeros((grid_pixel_number, grid_pixel_number))
    
    #Framerate
    clock = pygame.time.Clock()
    fps = 144
    #Time buffer
    time = 1/fps

    
    #Defining the font used later
    fontused = pygame.font.SysFont('Calibri', drawing_size//10, True, False)
    
    #Defining colors
    color_w = (255,255,255)
    color_g = (180,180,180)
    
    #Brush diameter
    brush_d = 3
    
    #Bool indicating if we should quit the pygame window
    done = False
    prob = model.predict_prob(np.transpose(grid))

    

    def draw_pixel(pos, color):
        c = max(min(255*color,255),0)
        pygame.draw.rect(screen,(c,c,c), pygame.Rect(pos[0], pos[1], grid_pixel_width, grid_pixel_width))
       
    def get_grid_position(pos):
        x = -1
        y = -1
        for i in range(grid_pixel_number):
            if pos[0] >= i * grid_pixel_width and pos[0] < (i + 1) * grid_pixel_width:
                x = i
            if pos[1] >= i * grid_pixel_width and pos[1] < (i + 1) * grid_pixel_width:
                y = i
        return (x, y)
    
    def set_color_around(pos, color, draw): 
        if brush_d%2==0:
            offset = 0.5
        else:
            offset = 0        
        r = brush_d//2+1
        for i in range(-r, r):
            for j in range(-r, r):
                ii = pos[0] + i
                jj = pos[1] + j
                d2 = 2*math.sqrt(((i + offset)**2 + (j + offset)**2)/(brush_d**2))        
                if ii >=0 and jj >= 0 and ii < grid_pixel_number and jj < grid_pixel_number and 1-d2 > 0:
                    if draw:
                        grid[ii, jj] = color
                    else:
                        grid_display_mouse_pos[ii, jj] = color
                    
    def draw_histogram(prob):
        max_prob = max(prob)
        #Draws 3 lines representing 25%, 50%, and 75% probability of being correct
        for i in range(1,4):
            pygame.draw.line(screen, color_g, (drawing_size, (i*bar_height)//4 ), 
                             (drawing_size + histogram_size, (i*bar_height)//4 ),1)
        #Draws the bars
        for i in range(10):
            pos = (drawing_size + bar_width*i, bar_height-10)
            pygame.draw.rect(screen,int_color(prob[i]/max_prob), pygame.Rect(pos[0], pos[1]-1, bar_width, -prob[i]*bar_height))
            screen.blit(fontused.render('{}'.format(i),True,(0,0,0)),(drawing_size*(1+1/40) + bar_width *i, bar_height))
    
    def reset_histogram():
        pygame.draw.rect(screen,color_w, pygame.Rect(drawing_size, 0, histogram_size, 10*drawing_size//9))
    
    def int_color(x):
        #Returns a color between green and red
        return (255*(1-x),255*x,0)

    
    #============================================
    #Main Loop
    #============================================
    while not done:
        
        clock.tick(fps)
        
        #Update the prediction
        #Change the multiplier to smoothen the writting or to update the prediction quicker
        time +=1
        if time>=0.1*fps:
            prob = model.predict_prob(np.transpose(grid))
            time = 0

        
        
        #Check for events
        for event in pygame.event.get():
           
            
            if event.type == pygame.QUIT:
                done = True       
             
            mp = pygame.mouse.get_pos()
            if mp[0] < drawing_size and mp[1] < drawing_size:
                #Left Click
                if pygame.mouse.get_pressed() == (1,0,0):
                    set_color_around(get_grid_position(mp),1,True)
        
                #Right click        
                elif pygame.mouse.get_pressed() == (0,0,1):
                    set_color_around(get_grid_position(mp),0,True)
                    grid_display_mouse_pos = np.zeros((grid_pixel_number, grid_pixel_number))
                    set_color_around(get_grid_position(mp),0.5,False)
                    
                else:
                    grid_display_mouse_pos = np.zeros((grid_pixel_number, grid_pixel_number))
                    set_color_around(get_grid_position(mp),1,False)
            
            #Middle click        
            if pygame.mouse.get_pressed() == (0,1,0) or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                grid = np.zeros((grid_pixel_number, grid_pixel_number))
                    
            #Mouse wheel
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    brush_d +=1
                    brush_d = min(brush_d,7) #7 chosen to be the max brush size
                elif event.button == 5:
                    brush_d -=1
                    brush_d = max(brush_d,1)
    

        #Print everything on screen
        
        #Draw the digit 
        pygame.draw.rect(screen,(0,0,0), pygame.Rect(0,0, drawing_size, drawing_size))
        for i in range(grid_pixel_number):
            for j in range(grid_pixel_number): 
                if grid[i,j]+grid_display_mouse_pos[i,j] >0:
                    draw_pixel((i * grid_pixel_width, j * grid_pixel_width), grid[i,j]+grid_display_mouse_pos[i,j]) 

        #Draw histogram
        reset_histogram()
        draw_histogram(prob)  
       

        pygame.display.flip() 
        
    pygame.quit()
