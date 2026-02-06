import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor

# ML Model: load or train automatically
MODEL_FILE = "dynamic_quantum_model.pkl"

def train_ml_model():
    st.info("Training dynamic quantum ML model... this may take a few seconds.")

    class Proc:
        def __init__(self, pid, burst):
            self.pid = pid
            self.burst_time = burst
            self.remaining_time = burst
            self.waiting_time = 0

    def simulate_rr(processes, quantum):
        processes = deepcopy(processes)
        timeline = []
        while any(p.remaining_time > 0 for p in processes):
            for p in [x for x in processes if x.remaining_time > 0]:
                run_time = min(quantum, p.remaining_time)
                p.remaining_time -= run_time
                timeline.extend([p.pid] * run_time)
                for other in processes:
                    if other != p and other.remaining_time > 0:
                        other.waiting_time += run_time

        avg_waiting = np.mean([p.waiting_time for p in processes])
        cpu_util = 100 * len([x for x in timeline if x > 0]) / max(1, len(timeline))
        return avg_waiting, cpu_util

    snapshots = []

    for _ in range(500):
        num_procs = np.random.randint(3, 8)
        bursts = np.random.randint(1, 11, size=num_procs)
        processes = [Proc(i + 1, bursts[i]) for i in range(num_procs)]

        num_ready = num_procs
        avg_remaining = np.mean(bursts)
        max_remaining = np.max(bursts)
        avg_waiting = 0
        cpu_util = 0

        best_q = 1
        best_wait = float("inf")

        for q in range(1, 11):
            aw, _ = simulate_rr(processes, q)
            if aw < best_wait:
                best_wait = aw
                best_q = q

        snapshots.append([
            num_ready,
            avg_remaining,
            max_remaining,
            avg_waiting,
            cpu_util,
            best_q
        ])

    df = pd.DataFrame(
        snapshots,
        columns=[
            "num_ready",
            "avg_remaining",
            "max_remaining",
            "avg_waiting",
            "cpu_util",
            "optimal_quantum"
        ]
    )

    X = df[["num_ready", "avg_remaining", "max_remaining", "avg_waiting", "cpu_util"]]
    y = df["optimal_quantum"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

    st.success("ML model trained and saved!")
    return model


if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = train_ml_model()


# Process class
class Process:
    def __init__(self, pid, arrival, burst, priority=1):
        self.pid = pid
        self.arrival_time = arrival
        self.burst_time = burst
        self.remaining_time = burst
        self.priority = priority
        self.state = "Ready"
        self.start_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.turnaround_time = 0


# Scheduling functions
def fcfs(ready_queue):
    return sorted(ready_queue, key=lambda x: x.arrival_time)

def sjf(ready_queue):
    return sorted(ready_queue, key=lambda x: x.burst_time)

def rr_get_next(ready_queue, last_index):
    if not ready_queue:
        return None, last_index
    next_index = (last_index + 1) % len(ready_queue)
    return ready_queue[next_index], next_index


# Streamlit UI
st.title("Process Manager Simulator")
st.sidebar.header("Simulation Settings")

algorithm = st.sidebar.selectbox("Scheduling Algorithm", ["FCFS", "SJF", "RR"])
sim_speed = st.sidebar.slider("Step Delay (s)", 0.05, 1.0, 0.2)
num_processes = st.sidebar.number_input("Initial number of processes", 1, 20, 5)
max_burst = st.sidebar.slider("Max burst time", 1, 20, 10)
max_priority = st.sidebar.slider("Max priority", 1, 5, 3)

start_sim = st.sidebar.button("Start Simulation")
add_proc_btn = st.sidebar.button("Add Random Process")


# Session state
if "process_list" not in st.session_state:
    st.session_state.process_list = []
if "cpu_timeline" not in st.session_state:
    st.session_state.cpu_timeline = []
if "current_time" not in st.session_state:
    st.session_state.current_time = 0
if "terminated_processes" not in st.session_state:
    st.session_state.terminated_processes = []
if "rr_index" not in st.session_state:
    st.session_state.rr_index = -1
if "deadlock" not in st.session_state:
    st.session_state.deadlock = False
if "current_quantum" not in st.session_state:
    st.session_state.current_quantum = 0


# Add random process
def add_random_process():
    new_pid = len(st.session_state.process_list) + 1
    arrival = st.session_state.current_time
    burst = np.random.randint(1, max_burst + 1)
    priority = np.random.randint(1, max_priority + 1)
    st.session_state.process_list.append(Process(new_pid, arrival, burst, priority))
    st.session_state.deadlock = False


if add_proc_btn:
    add_random_process()


# Reset simulation if all terminated
if start_sim:
    if st.session_state.process_list and all(p.state == "Terminated" for p in st.session_state.process_list):
        st.session_state.process_list = []
        st.session_state.cpu_timeline = []
        st.session_state.current_time = 0
        st.session_state.terminated_processes = []
        st.session_state.rr_index = -1
        st.session_state.deadlock = False
        st.session_state.current_quantum = 0

    if len(st.session_state.process_list) == 0:
        for i in range(num_processes):
            arrival = np.random.randint(0, 5)
            burst = np.random.randint(1, max_burst + 1)
            priority = np.random.randint(1, max_priority + 1)
            st.session_state.process_list.append(Process(i + 1, arrival, burst, priority))


# Simulation loop
placeholder = st.empty()

while st.session_state.process_list and not all(p.state == "Terminated" for p in st.session_state.process_list):
    ready_queue = [
        p for p in st.session_state.process_list
        if p.arrival_time <= st.session_state.current_time and p.state != "Terminated"
    ]

    if ready_queue:
        if algorithm == "FCFS":
            current = fcfs(ready_queue)[0]
            run_time = 1
        elif algorithm == "SJF":
            current = sjf(ready_queue)[0]
            run_time = 1
        else:
            current, st.session_state.rr_index = rr_get_next(ready_queue, st.session_state.rr_index)

            num_ready = len(ready_queue)
            avg_remaining = np.mean([p.remaining_time for p in ready_queue])
            max_remaining = np.max([p.remaining_time for p in ready_queue])
            avg_waiting = np.mean([p.waiting_time for p in ready_queue])
            cpu_util = len([x for x in st.session_state.cpu_timeline if x > 0]) / max(1, len(st.session_state.cpu_timeline))

            features = [[num_ready, avg_remaining, max_remaining, avg_waiting, cpu_util]]
            time_quantum = max(1, int(model.predict(features)[0]))
            st.session_state.current_quantum = time_quantum
            run_time = min(time_quantum, current.remaining_time)

        current.state = "Running"
        if current.start_time is None:
            current.start_time = st.session_state.current_time

        for _ in range(run_time):
            current.remaining_time -= 1
            st.session_state.cpu_timeline.append(current.pid)
            for p in ready_queue:
                if p != current:
                    p.waiting_time += 1
            st.session_state.current_time += 1
            time.sleep(sim_speed)

        if current.remaining_time == 0:
            current.state = "Terminated"
            current.finish_time = st.session_state.current_time
            current.turnaround_time = current.finish_time - current.arrival_time
            st.session_state.terminated_processes.append(current)

    else:
        st.session_state.cpu_timeline.append(0)
        st.session_state.current_time += 1
        time.sleep(sim_speed)

    # Display
    with placeholder.container():
        if algorithm == "RR" and ready_queue:
            st.markdown(f"Current Dynamic RR Time Quantum: {st.session_state.current_quantum}")

        df = pd.DataFrame([{
            "PID": p.pid,
            "State": p.state,
            "Arrival": p.arrival_time,
            "Burst": p.burst_time,
            "Remaining": p.remaining_time,
            "Priority": p.priority,
            "Waiting": p.waiting_time
        } for p in st.session_state.process_list])

        st.dataframe(df)
        st.bar_chart(pd.DataFrame(st.session_state.cpu_timeline, columns=["PID"]))

        terminated = [p for p in st.session_state.process_list if p.state == "Terminated"]
        if terminated:
            avg_waiting = np.mean([p.waiting_time for p in terminated])
            avg_turnaround = np.mean([p.turnaround_time for p in terminated])
            cpu_util = 100 * len([x for x in st.session_state.cpu_timeline if x > 0]) / max(1, len(st.session_state.cpu_timeline))

            st.markdown(f"Average Waiting Time: {avg_waiting:.2f}")
            st.markdown(f"Average Turnaround Time: {avg_turnaround:.2f}")
            st.markdown(f"CPU Utilization: {cpu_util:.2f}%")
