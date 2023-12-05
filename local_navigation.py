MAX_SPEED_ALLOWED=150
WALL_THRESHOLD=500


@tdmclient.notebook.sync_var
def get_prox():
    global prox_horizontal
    return prox_horizontal


#local navigation function
def local_navigation():
    global prox_horizontal, motor_left_target, motor_right_target,motor_left_speed,motor_right_speed
    while max(get_prox())>WALL_THRESHOLD:
        weight_left = [25,  15, -15, -15, -25]
        weight_right = [-25, -15, -15,  15,  25]
    
        # Scale factors for sensors
        sensor_scale = 500
    
        mem_sensor = [0,0,0,0,0]
        prox_horizontal = get_prox()
        
        for i in range(5):
            # Get and scale inputs
            mem_sensor[i] = prox_horizontal[i]//sensor_scale
    
        y = [motor_left_speed,motor_right_speed]   
        
        for i in range(len(mem_sensor)):   
            # Compute outputs of neurons and set motor powers
            y[0] = y[0] + mem_sensor[i] * weight_left[i]
            y[1] = y[1] + mem_sensor[i] * weight_right[i]
    
        # Set motor powers
        set_var(motor_left_target = min(y[0],MAX_SPEED_ALLOWED))
        set_var(motor_right_target = min(y[1],MAX_SPEED_ALLOWED))
        time.sleep(0.2)
    return