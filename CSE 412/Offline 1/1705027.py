import numpy as np
from queue import Queue

# single server queue simulation
queue_limit = 100
ARRIVAL_EVENT = 1
DEPARTURE_EVENT = 2
SERVER_BUSY = 1
SERVER_IDLE = 0

def expon(mean):
    uniform_random = np.random.uniform(0, 1)
    return -mean * np.log(uniform_random)

def initialize():
    global server_queue, simulation_time, number_in_queue, server_status, time_last_event, \
        total_of_delays, area_number_in_queue, area_server_status, number_of_customer_delays, \
            time_next_arrival_event, time_next_departure_event, number_of_customers_arrived

    # server queue
    server_queue = Queue(maxsize=queue_limit)

    # initialize the simulation clock
    simulation_time = 0.0

    # Initialize the state variables.
    number_in_queue = 0
    server_status = SERVER_IDLE
    time_last_event = 0.0
    number_of_customers_arrived = 0

    # Initialize the statistical counters.
    total_of_delays = 0.0
    number_of_customer_delays = 0
    area_number_in_queue = 0.0
    area_server_status = 0.0

    # Initialize first event. Since no customers are present, the departure (service completion) event is eliminated from consideration.

    time_next_arrival_event = simulation_time + expon(mean_interval_time)
    number_of_customers_arrived = 1
    time_next_departure_event = np.inf


def timing():
    global simulation_time
    
    if time_next_arrival_event < time_next_departure_event:
        simulation_time = time_next_arrival_event
        event = ARRIVAL_EVENT
    else:
        simulation_time = time_next_departure_event
        event = DEPARTURE_EVENT
    return event

def update_time_avg_stats():
    global time_last_event, simulation_time, area_number_in_queue, area_server_status
    # compute time since last event, and update last-event-time marker
    time_since_last_event = simulation_time - time_last_event
    time_last_event = simulation_time

    # update area under number-in-queue function
    area_number_in_queue += number_in_queue * time_since_last_event

    # update area under server-busy indicator function
    area_server_status += server_status * time_since_last_event

def arrive():
    global time_next_arrival_event, time_next_departure_event, number_of_customers_arrived, \
        number_in_queue, total_of_delays, number_of_customer_delays, server_queue, server_status
    # schedule the next arrival
    if number_of_customers_arrived < number_of_customers:
        time_next_arrival_event = simulation_time + expon(mean_interval_time)
        number_of_customers_arrived += 1
    else:
        time_next_arrival_event = np.inf
    
    # check if the server is busy
    if server_status == SERVER_BUSY:
        # server is busy, so increment the number of customers in queue
        number_in_queue += 1
        # check if the queue is full
        if number_in_queue > queue_limit:
            print("Queue overflow")
            exit(1)
        # add the customer to the queue
        server_queue.put(simulation_time)
    else:
        # server is idle, so arriving customer has a delay of zero
        delay = 0.0
        total_of_delays += delay

        # increment the number of customers delayed, and make server busy
        number_of_customer_delays += 1
        server_status = SERVER_BUSY

        # schedule a departure (service completion)
        time_next_departure_event = simulation_time + expon(mean_service_time)


def depart():
    global server_status, time_next_departure_event, total_of_delays, number_in_queue, \
        number_of_customer_delays, server_queue, number_of_customers_arrived
    # check to see whether queue is empty
    if number_in_queue == 0:
        # The queue is empty so make the server idle and 
        # eliminate the departure (service completion) event from consideration.
        server_status = SERVER_IDLE
        time_next_departure_event = np.inf
    else:
        # The queue is nonempty, so decrement the number of customers in queue.
        number_in_queue -= 1
        # compute the delay of the customer who is beginning service and update the total delay accumulator.
        delay = simulation_time - server_queue.get()
        total_of_delays += delay

        # Increment the number of customers delayed, and schedule departure.
        number_of_customer_delays += 1
        time_next_departure_event = simulation_time + expon(mean_service_time)

def report():
    global total_of_delays, number_of_customer_delays, area_number_in_queue, area_server_status, \
        simulation_time, number_of_customers_arrived, mean_interval_time, mean_service_time, \
            number_of_customers

    # compute and write estimates of desired measures of performance
    average_delay = total_of_delays / number_of_customer_delays
    average_number_in_queue = area_number_in_queue / simulation_time
    average_server_status = area_server_status / simulation_time

    with open('output.txt', 'w') as f:
        f.write("Average delay in queue: {} minutes\n".format(average_delay))
        f.write("Average number in queue: {}\n".format(average_number_in_queue))
        f.write("Server utilization: {}\n".format(average_server_status))
        f.write('Time simulation ended: {} minutes\n'.format(simulation_time))

def main():
    global mean_interval_time, mean_service_time, number_of_customers
    
    with open('input.txt') as f:
        lines = f.read().split("\n")
    if len(lines) == 3:
        mean_interval_time = float(lines[0])
        mean_service_time = float(lines[1])
        number_of_customers = int(lines[2])
    else:
        print("Invalid input")
        return

    # initialize the simulation
    initialize()

    # run the simulation while more delays are still needed
    while number_of_customer_delays < number_of_customers:
        # determine the next event
        event = timing()

        # update time-average statistical accumulators
        update_time_avg_stats()

        # arrival event
        if event == ARRIVAL_EVENT:
            arrive()
        # departure event
        elif event == DEPARTURE_EVENT:
            depart()
        
    # Invoke the report generator and end the simulation
    report()

if __name__=="__main__":
    main()
