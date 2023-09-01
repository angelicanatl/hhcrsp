from instance import Instance, Patient
import re
from collections import defaultdict
from typing import Callable, Any, TextIO, Iterable
import numpy as np
import math
from itertools import permutations
import json

try:
    import graphviz
    graphviz_available = True
except ImportError:
    graphviz_available = False

class Service:
    def __init__(self, patient : Patient, caregiver : int, service_type : int, start_time : float , end_time : float):
        assert type(patient) == Patient
        self.patient = patient
        self.caregiver = caregiver
        self.service_type = service_type
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self) -> str:
        return f"p={self.patient} c={self.caregiver} t={self.service_type} [{self.start_time}-{self.end_time}]"

class Solution:
    def __init__(self, content : str | TextIO, instance : Instance, scale_integers : int = 1):
        if type(content) == TextIO:
            content = content.read()
        if re.search(r'\{', content):
            self.dispatch_json(json.loads(content), instance , scale_integers)
        else:
            self.parse_content_el(content, instance, scale_integers)

    def dispatch_json(self, content : str, instance : Instance, scale_integers : int = 1) -> int:
        self.routes = defaultdict(list)
        self.served_patients = defaultdict(list)
        for i, r in enumerate(content['routes']):
            caregiver = r.get('caregiver') or r.get('caregiver_id')
            self.routes[caregiver] = []
            for step in r.get('locations', []):  
                patient = next(filter(lambda p: (p.id == step.get('patient')) or (p.id == step.get('patient_id')), instance.patients.values()))
                service = Service(patient, caregiver, step.get('service') or step.get('service_id'), step['arrival_time'], step['departure_time'])
                self.routes[caregiver].append(service)
                self.served_patients[patient.id].append(service) 
        for p, sp in self.served_patients.items():
            if len(sp) == 1:
                continue
            if sp[0].service_type == instance.patients[p].required_services[0]:
                continue
            sp[0], sp[1] = sp[1], sp[0]

    def parse_content_kummer(self, content : str, instance : Instance):
        def read_matrix(content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = r'(\d+)\s*(\d+)'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid: expecting {pattern} at line {counter}"
            vehicle = int(m.group(1))
            length = int(m.group(2))            
            content.pop(0)

            res = np.array([
                [conv(v) for v in re.split(r'\s+', content[i]) if v.strip()] 
                for i in range(length)
            ])
            for _ in range(length):
                content.pop(0)            
            return counter + 1 + length, vehicle, res
        lines = list(filter(lambda l: not l.startswith('#') and l.strip(), content.split('\n')))
        self.routes = defaultdict(list)
        self.served_patients = defaultdict(list)
        l = 1
        while lines:
            l, caregiver, route = read_matrix(lines, l, int)
            assert np.all(route[:,2] == caregiver)
            prev_node = 0 # start at depot
            departing_time = 0
            for step in route[:-1]:        
                p = instance.patients[step[1]]
                start_time = max(departing_time + instance.travel_distance[prev_node, step[1]], p.time_window[0])
                departing_time = start_time + p.processing_times[step[3]]
                prev_node = step[1]
                patient = instance.patients[step[1] - 1]
                service = Service(patient, caregiver, step[3], start_time, departing_time)
                self.routes[caregiver].append(service)
                self.served_patients[patient.id].append(service)
        changes = True
        while changes:
            changes = False
            for caregiver, route in self.routes.items():
                push_quantity = 0
                for i in range(len(route)):
                    step = route[i]
                    p = instance.patients[step.patient]
                    step.start_time += push_quantity
                    step.end_time += push_quantity
                    if len(p.required_services) == 1:
                        continue
                    # TODO: it is based on the assumption of at most pairs of service                    
                    other_service = next(filter(lambda s: s.service_type != step.service_type, self.served_patients[p.id]))
                    # push the service for minimum distance if it's the second
                    key = (other_service.service_type, step.service_type)
                    if key in p.service_sync_distance:
                        min_distance = p.service_sync_distance[key][0]
                        if step.start_time < other_service.start_time + min_distance:
                            changed = True
                            q = other_service.start_time + min_distance - step.start_time
                            step.start_time += q
                            step.end_time += q
                            push_quantity += q
                    # push the service for maximum distance if it's the first
                    key = (step.service_type, other_service.service_type)
                    if key in p.service_sync_distance:
                        max_distance = p.service_sync_distance[key][1]
                        if other_service.start_time > step.start_time + max_distance:
                            changed = True
                            q = other_service.start_time - (step.start_time + max_distance)
                            step.start_time += q
                            step.end_time += q
                            push_quantity += q

    def parse_content_el(self, content : str, instance: Instance, scale_integers : int = 1):
        lines = content.split('\n')
        m = re.match(r'Global order: (?:\d+\s*)+', lines[0])
        assert m
        lines.pop(0)
        m = re.match(r'Routes:', lines[0])
        lines.pop(0)
        self.routes = defaultdict(list)
        self.served_patients = defaultdict(list)
        while True:
            m = re.match('(\d+)([:#])', lines[0])
            if not m:
                break
            c = int(m.group(1))
            if m.group(2) == "#":
                self.routes[c] = []
                lines.pop(0)
                continue
            data = list(map(lambda t: tuple(re.split(r'[/\-\s]', t.strip())), lines[0].rstrip(';').split(':')[1].split(',')))
            for d in data:
                p = int(d[0]) + 1
                patient = instance.patients[p - 1]
                s = Service(patient, c, int(d[1]), float(d[2]) / scale_integers, float(d[3]) / scale_integers)
                self.routes[c].append(s)
                self.served_patients[p.id].append(s)
            lines.pop(0)        

    def to_kummer(self):
        content = ["# dummy line"] * 4
        for r, route in enumerate(self.routes.values()):
            content.append(f"{r} {len(route) + 1}")
            previous_patient = 0
            for step in route:
                content.append(f"{previous_patient} {step.patient} {r} {step.service_type}")
                previous_patient = step.patient
            content.append(f"{previous_patient} 0 {r} 0")
        return "\n".join(content)


    def to_graph(self, instance : Instance, scale : float = 0.5):
        if graphviz_available:
            COLORS = ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"]
            g = graphviz.Digraph(comment='Home Healthcare RSP')
            g.graph_attr['layout'] = 'neato'
            g.node('0', 'd_0',  pos=f'{instance.depot_location[0] * scale},{instance.depot_location[1] * scale}!', color='lightgray', style='filled', shape='rect')
            for p in instance.patients.values():
                g.node(f'{p.index}', f'{p.index}', pos=f'{p.location[0] * scale},{p.location[1] * scale}!')
            
            for c, route in self.routes.items():
                current_node = 0
                for step in route:
                    g.edge(f'{current_node}', f'{step.patient}', color=COLORS[c % len(COLORS)], label=f'Caregiver: {c}')
                    current_node = step.patient
                g.edge(f'{current_node}', f'{0}', color=COLORS[c % len(COLORS)], label=f'Caregiver: {c}')
            # for i, j in combinations(range(len(self.patients) + 1), 2):
            #     g.edge(f'{i}', f'{j}', label=f'{self.travel_distance[i][j]:.1f}')
            g.view()

def validate_solution(instance : Instance, solution : Solution, route_tardiness : bool = True) -> dict:
    assert len(solution.routes) == instance.caregivers, f"Expected {instance.caregivers} routes (one for each caregiver), found {len(solution.routes)}"
    assert len(solution.served_patients) == len(instance.patients), f"Expected {len(instance.patients)} served patients, found {len(solution.served_patients)}"

    cost = {}
    cost['distance_traveled'] = 0.0
    cost['tardiness'] = {}
    if route_tardiness:
        cost['route_tardiness'] = {}
    feasible = True
    for c, route in solution.routes.items():
        departing_time = 0
        previous_patient = 0        
        for service in route:
            p = instance.patients[service.patient.id]
            assert service.service_type in instance.skills[c], f"Caregiver {c} providing service {service.service_type} at patient {service.patient} which is not in his/her skills {instance.skills[c]}"
            assert service.service_type in p.required_services, f"Caregiver {c} providing service {service.service_type} at patient {service.patient} which is not among those required by the patient {p.required_services}"
            assert math.isclose(service.end_time - service.start_time, p.processing_times[service.service_type], abs_tol=1E-3), f"Processing time {service.end_time - service.start_time} for service {service.service_type} at patient {service.patient} not matching with the expected value {p.processing_times[service.service_type]}"
            distance = instance.travel_distance[previous_patient, service.patient.index]
            assert service.start_time >= departing_time + distance or math.isclose(service.start_time, departing_time + distance, abs_tol=1E-3), f"Start service time of caregiver {c} at patient {service.patient} is too early because of traveling distance {distance} and departing time {departing_time}, estimated at least {departing_time + distance}, found {service.start_time}"
            cost['distance_traveled'] += instance.travel_distance[previous_patient, service.patient.index]
            cost['tardiness'][str((p.index, service.service_type))] = max(service.start_time - p.time_window[1], 0.0)
            departing_time = service.end_time
            previous_patient = p.index
        arrival_at_depot = departing_time + instance.travel_distance[previous_patient, 0]
        cost['distance_traveled'] += instance.travel_distance[previous_patient, 0]
        if route_tardiness:
            cost['route_tardiness'][c] = max(arrival_at_depot - instance.depot_time_window[1], 0.0)
    
    for p_index, service in solution.served_patients.items():
        p = instance.patients[p_index]        
        if len(p.required_services) == 1:
            assert not p.service_sync_distance, "Error, specified sync distance for single service patient {p}"
            continue
        for s_pair in permutations(service):
            key = tuple(map(lambda s: s.service_type, s_pair))
            if key in p.service_sync_distance:
                assert s_pair[1].start_time >= s_pair[0].start_time + p.service_sync_distance[key][0] or math.isclose(s_pair[1].start_time, s_pair[0].start_time + p.service_sync_distance[key][0], rel_tol=1E-3), f"Service {s_pair[1]} starts too early with respect to service {s_pair[0]}, it should be at least {p.service_sync_distance[key][0]} apart"
                assert s_pair[1].start_time <= s_pair[0].start_time + p.service_sync_distance[key][1] or math.isclose(s_pair[1].start_time, s_pair[0].start_time + p.service_sync_distance[key][1], rel_tol=1E-3), f"Service {s_pair[1]} starts too late with respect to service {s_pair[0]}, it should be at most {p.service_sync_distance[1]} apart"
        for s in service:
            assert p.time_window[0] <= s.start_time or math.isclose(p.time_window[0], s.start_time, rel_tol=1E-3), f"Service {s} starts too early for patient time window {p.time_window}"    

    # computation of cost
    if route_tardiness:
        cost['max_tardiness'] = max([*cost['tardiness'].values(), *cost['route_tardiness'].values()])
        cost['total_tardiness'] = sum([*cost['tardiness'].values(), *cost['route_tardiness'].values()])
    else:
        cost['max_tardiness'] = max(cost['tardiness'].values())
        cost['total_tardiness'] = sum(cost['tardiness'].values())
    cost['total_cost'] = cost['distance_traveled'] / 3 + cost['total_tardiness'] / 3 + cost['max_tardiness'] / 3

    return cost
