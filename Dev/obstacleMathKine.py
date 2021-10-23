from inverseKine import NLinkArm
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)

def get_random_goal():
    SAREA = 4.0
    x=random.randint(-20,20)/10
    y=random.randint(-20,-10)/10
    #return [SAREA * random() - SAREA / 2.0,SAREA * random() - SAREA / 2.0]
    return [x,y]

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)

def get_distance(p1,p2): #measure distance between points
    y1,y2=p1[1],p2[1]
    x1,x2=p1[0],p2[0]
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False

# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 2
N_ITERATIONS = 10000 #more solutions = more accurate

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True

link_lengths = [1] * N_LINKS
joint_angles = np.array([90] * N_LINKS)
goal_pos = get_random_goal()
arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
state = WAIT_FOR_NEW_GOAL
solution_found = False

i_goal = 0
while True:
    grad=random.randint(-50,30)/100 #generate random
    c=-3
    
    if grad<0: c=3
    graph(str(grad)+'*x+'+str(c), range(-10, 40)) #draw obstacle
    
    old_goal = np.array(goal_pos)
    goal_pos = np.array(arm.goal)
    end_effector = arm.end_effector
    errors, distance = distance_to_goal(end_effector, goal_pos)
    if state is WAIT_FOR_NEW_GOAL:

            if distance > 0.1 and not solution_found:
                joint_goal_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, goal_pos)
                print(joint_goal_angles)
                if not solution_found:
                    print("Solution could not be found.")
                    state = WAIT_FOR_NEW_GOAL
                    arm.goal = get_random_goal()
                elif solution_found:
                    state = MOVING_TO_GOAL
    elif state is MOVING_TO_GOAL:
            if distance > 0.1 and all(old_goal == goal_pos):
                joint_angles = joint_angles + Kp * \
                    ang_diff(joint_goal_angles, joint_angles) * dt
            else:
                state = WAIT_FOR_NEW_GOAL
                solution_found = False
                arm.goal = get_random_goal()
                i_goal += 1
    if i_goal >= 5:
            break
    arm.update_joints(joint_angles)
