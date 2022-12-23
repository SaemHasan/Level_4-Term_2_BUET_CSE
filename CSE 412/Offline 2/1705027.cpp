#include <iostream>
#include <cmath>
#include "lcgrand.h"

#define NUMBER_OF_EVENTS 4
#define ORDER_ARRIVAL 1
#define DEMAND_FOR_PRODUCT 2
#define END_OF_SIMULATION 3
#define INVENTORY_EVALUATION 4

using namespace std;

int small_s, big_S, order_amount;
int initial_inventory_level, number_of_months, number_of_policies, number_of_values_demand;
double mean_interdemand, setupCost, incrementalCost, holdingCost, shortageCost, minlag, maxlag;

double *prob_distrib_demand;
double time_next_event[NUMBER_OF_EVENTS + 1]; // 1: arrival 2: demand
                                              // 3: end of simulation 4: inventory evaluation
double simulation_time, time_last_event, total_ordering_cost, area_holding, area_shortage;
int inventory_level, next_event_type;

void print();
void initialize();
double expon(double mean);
void timing();
void update_time_avg_stats();
void order_arrival();
void demand_for_product();
int generate_demand();
void inventory_evaluation();
double uniform(double a, double b);
void report();

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    cin >> initial_inventory_level >> number_of_months >> number_of_policies >> number_of_values_demand;
    cin >> mean_interdemand >> setupCost >> incrementalCost >> holdingCost >> shortageCost >> minlag >> maxlag;

    prob_distrib_demand = new double[number_of_values_demand+1];
    for (int i = 1; i <= number_of_values_demand; ++i)
    {
        cin >> prob_distrib_demand[i];
    }
    print();

    for (int i = 0; i < number_of_policies; i++)
    {
        cin >> small_s >> big_S;
        initialize();

        do
        {
            // determine the next event
            timing();
            // update time-average statistical accumulators
            update_time_avg_stats();

            // invoke the appropriate event function
            // cout << "next_event_type: " << next_event_type << endl;
            switch (next_event_type)
            {
            case ORDER_ARRIVAL:
                order_arrival();
                break;
            case DEMAND_FOR_PRODUCT:
                demand_for_product();
                break;
            case INVENTORY_EVALUATION:
                inventory_evaluation();
                break;
            case END_OF_SIMULATION:
                report();
                break;
            }

        } while (next_event_type != END_OF_SIMULATION);
    }
}

void initialize()
{
    // simulation clock
    simulation_time = 0.0;
    // initialize the state variables
    inventory_level = initial_inventory_level;
    time_last_event = 0.0;

    // initialize the statistical counters
    total_ordering_cost = 0.0;
    area_holding = 0.0;
    area_shortage = 0.0;

    /* Initialize the event list.  Since no order is outstanding, the order-
       arrival event is eliminated from consideration. */

    time_next_event[ORDER_ARRIVAL] = 1.0e+30;
    time_next_event[DEMAND_FOR_PRODUCT] = simulation_time + expon(mean_interdemand);
    time_next_event[END_OF_SIMULATION] = number_of_months;
    time_next_event[INVENTORY_EVALUATION] = 0.0;
}

void timing()
{
    double min_time_next_event = 1.0e+29;
    next_event_type = 0;

    for (int i = 1; i <= NUMBER_OF_EVENTS; i++)
    {
        if (time_next_event[i] < min_time_next_event)
        {
            min_time_next_event = time_next_event[i];
            next_event_type = i;
        }
    }

    if (next_event_type == 0)
    {
        cout << "Event list empty at time " << simulation_time << endl;
        exit(1);
    }

    // The event has been determined; advance the simulation clock.
    simulation_time = min_time_next_event;
}

void update_time_avg_stats(){
    double time_since_last_event;
    time_since_last_event = simulation_time - time_last_event;
    time_last_event = simulation_time;
    
    if (inventory_level > 0)
        area_holding += inventory_level * time_since_last_event;
    else
        area_shortage -= inventory_level * time_since_last_event;
}

void order_arrival(){
    inventory_level += order_amount;
    /* Since no order is now outstanding, eliminate the order-arrival event from
       consideration. */
    time_next_event[ORDER_ARRIVAL] = 1.0e+30;
}

void demand_for_product(){
    /* Decrement the inventory level by a generated demand size. */
    inventory_level -= generate_demand();
    /* Schedule the next demand event. */
    time_next_event[DEMAND_FOR_PRODUCT] = simulation_time + expon(mean_interdemand);
}

void inventory_evaluation(){
    if (inventory_level < small_s)
    {
        order_amount = big_S - inventory_level;
        total_ordering_cost += setupCost + incrementalCost * order_amount;

        time_next_event[ORDER_ARRIVAL] = simulation_time + uniform(minlag, maxlag);
    }
    time_next_event[INVENTORY_EVALUATION] = simulation_time + 1.0;
}

void report(){
    // cout << "simulation time : " << simulation_time << endl;
    // cout << "area holding :  "<<area_holding << " " << area_shortage << endl;
    // cout <<
    double average_holding_cost, average_shortage_cost, average_ordering_cost, total_cost;
    average_holding_cost = holdingCost * area_holding / number_of_months;
    average_shortage_cost = shortageCost * area_shortage / number_of_months;
    average_ordering_cost = total_ordering_cost / number_of_months;
    total_cost = average_holding_cost + average_shortage_cost + average_ordering_cost;
    cout << "(" << small_s << ", " << big_S << ")\t\t" << total_cost << "\t\t\t"<<average_ordering_cost<<
        "\t\t\t"<<average_holding_cost<<"\t\t\t"<<average_shortage_cost<<"\n\n";
    
    // cout << "s: " << small_s << " S: " << big_S << endl;
    // cout << "Average holding cost: " << average_holding_cost << endl;
    // cout << "Average shortage cost: " << average_shortage_cost << endl;
    // cout << "Average ordering cost: " << average_ordering_cost << endl;
    // cout<<endl;
}

void print()
{
    // cout << "lc grand "<< lcgrand(1) << endl;
    cout << "Single product inventory system\n\n";
    cout << "Initial Inventory Level "<< initial_inventory_level << " items\n\n";
    cout << "Number of demand sizes " << number_of_values_demand << "\n\n";
    cout << "Distribution function for demand sizes\t";
    for (int i = 1; i <= number_of_values_demand; i++)
    {
        cout << prob_distrib_demand[i] << " ";
    }
    cout << "\n\n";
    cout << "Mean interdemand time " << mean_interdemand << " months\n\n";
    cout << "Delivery lag range " << minlag << " to " << maxlag << " months\n\n";
    cout << "Length of the simulation " << number_of_months << " months\n\n";
    cout << "K = " << setupCost << " i = " << incrementalCost << " h = " << holdingCost << " pi = " << shortageCost << endl<<endl;
    cout << "Number of policies " << number_of_policies << "\n\n";
    cout << "               Average         Average         Average         Average\n";
    cout << "Policy        total cost    ordering cost   holding cost    shortage cost\n";
    cout << "------        ----------    -------------   -------------   -------------\n";

    // cout << initial_inventory_level << " " << number_of_months << " " << number_of_policies << " " << number_of_values_demand << endl;
    // cout << mean_interdemand << " " << setupCost << " " << incrementalCost << " " << holdingCost << " " << shortageCost << " " << minlag << " " << maxlag << endl;
    // for (int i = 0; i < number_of_values_demand; i++)
    // {
    //     cout << prob_distrib_demand[i] << " ";
    // }
    // cout << endl;
}

int generate_demand(){
    int i;
    double u = lcgrand(1);
    /* Return a random integer in accordance with the (cumulative) distribution
       function prob_distrib. */
    
    for(i=1; u>=prob_distrib_demand[i]; ++i);

    return i;
}

double expon(double mean)
{
    return -mean * log(lcgrand(1));
}

double uniform(double a, double b)
{
    return a + (b - a) * lcgrand(1);
}