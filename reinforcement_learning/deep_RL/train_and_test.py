import time
import numpy as np

from random_environment import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":
    greedy_path = []
    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)
    #print(random_seed)
    #np.random.seed(1606404710)
    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600
    epi = 0
    # Train the agent, until the time is up
    while time.time() < end_time:
        epi += 1
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        if epi %100==0:
            greedy_path = []

            greedy_state = environment.init_state
            has_reached_goal = False
            for step_num in range(100):
                action = agent.get_greedy_action(greedy_state)
                next_greedy_state, distance_to_goal = environment.step(greedy_state, action)
                # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
                greedy_path.append([greedy_state,next_greedy_state])
                if distance_to_goal < 0.03:
                    has_reached_goal = True
                    break
                greedy_state = next_greedy_state
            environment.update_greedy_path(greedy_path)

        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            environment.show(state)
    print('total episode:',epi)
    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
    environment.save()
