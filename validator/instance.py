from pyparsing import Word, alphas, nums, alphanums, empty, \
    Literal, delimitedList, Group, Combine, FollowedBy, Optional, Forward, OneOrMore, restOfLine, \
    Dict
import numpy as np
import re
from typing import Callable, Any, TextIO
from itertools import combinations
from collections import defaultdict
from scipy import stats as st
try:
    import graphviz
    graphviz_available = True
except ImportError:
    graphviz_available = False
import json
import sys

class Patient:
    """
    A patient is represented by an index (to be used in the distance matrix), requires a set of skills and is allowed 
    to be served within a given time window. In case it is relevant there is a requirement of synchronization distance
    and for each required service he/she has a processing time.
    """
    def __init__(self, index : int, id : str | None = None):
        self.id = id or f"p{index}"
        self.index = index
        self.required_services = set()
        self.time_window = (0.0, float('inf'))
        self.service_sync_distance = {}
        self.processing_times = dict()
        self.location = None

    def __repr__(self) -> str:
        return f"Patient {self.index}: s={self.required_services}, t={self.time_window}, d={self.service_sync_distance}, p={self.processing_times}"

class Instance:
    def __init__(self, content : str | TextIO):
        if type(content) == TextIO:
            content = content.read()
        if content.strip().startswith('{'):
            self.dispatch_json(json.loads(content))
        elif re.search(r'nbNodes\n\d+', "\n".join(content.split('\n')[:2]), re.MULTILINE):
            self.parse_kummer(content)
        elif re.search(r'e\n(\d+\s*)+\n', content, re.MULTILINE):       
            self.parse_vns(content)
        else:
            self.parse_cplex(content)

    def parse_vns(self, content : str):
        def read_scalar(name: str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}\\s*=\\s*(\\d+)\\s*;'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid at line {counter}, expecting {pattern}"
            content.pop(0)
            return counter + 1, conv(m.group(1))
        def read_array(name : str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}\\s*'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid: expecting {pattern} at line {counter}"
            m = re.match(r'\d+\s+', content[1])
            assert m, f"File format not valid: expecting array {name} at line {counter + 1}"
            res = np.array([conv(v) for v in re.split(r'\s+', content[1]) if v.strip()])
            content.pop(0)
            content.pop(0)            
            return counter + 2, res
        def read_matrix(name : str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}\\s*'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid: expecting {pattern} at line {counter}"
            content.pop(0)
            l = 1
            res = []
            while True:
                m = re.match(r'\d+\s+', content[0])
                if not m:
                    break
                res.append([conv(v) for v in re.split(r'\s+', content[0]) if v.strip()])
                l += 1
                content.pop(0)
            return counter + l + 1, np.array(res)

        data = {}
        # this is a very nasty format
        lines = content.split('\n')
        l = 1
        l, data['services'] = read_scalar('nbServi', lines, l, int)
        l, data['vehicles'] = read_scalar('nbVehi', lines, l, int)
        l, data['customers'] = read_scalar('nbCust', lines, l, int)
        l, data['synch'] = read_scalar('nbSynch', lines, l, int)
        l, data['e'] = read_array('e', lines, l, int)
        l, data['l'] = read_array('l', lines, l, int)
        l, data['cx'] = read_array('cx', lines, l, int)
        l, data['cy'] = read_array('cy', lines, l, int)
        l, data['s'] = read_matrix('s', lines, l, int)
        l, data['delta'] = read_matrix('delta', lines, l, int)
        l, data['p'] = read_matrix('p', lines, l, int)
        l, data['att'] = read_matrix('att', lines, l, int)
        self.depots = {0}
        self.patients = {}
        for i in range(data['customers'] + data['synch']):
            patient = Patient(i + 1)
            self.patients[patient.id] = patient
        self.caregivers = data['vehicles']         
        self.service_types = data['services']    
        self.skills = { i: set() for i in range(self.caregivers) }
        assert data['att'].shape[0] == self.caregivers and data['att'].shape[1] == self.service_types
        for c in range(self.caregivers):
            self.skills[c] = set(np.flatnonzero(data['att'][c] > 0))
        self.required_services = {}
        assert data['s'].shape[0] == len(self.patients) + len(self.depots) and data['s'].shape[1] == self.service_types        
        assert data['e'].shape[0] == len(self.patients) + len(self.depots)
        assert data['l'].shape[0] == len(self.patients) + len(self.depots)
        assert data['delta'].shape[0] == len(self.patients) + len(self.depots) and data['delta'].shape[1] == 2
        data['p'] = data['p'].reshape((len(self.patients) + len(self.depots), self.caregivers, self.service_types))
        assert data['p'].shape[0] == len(self.patients) + len(self.depots) and data['p'].shape[1] == self.caregivers and data['p'].shape[2] == self.service_types
        if 'cx' in data and 'cy' in data:
            assert data['cx'].shape[0] == len(self.patients) + len(self.depots) and data['cy'].shape[0] == len(self.patients) + len(self.depots)
        for p in self.patients.values():
            p.required_services = set(np.flatnonzero(data['s'][p.index] > 0))
            assert len(p.required_services) <= 2
            p.time_window = (data['e'][p.index], data['l'][p.index])
            if len(p.required_services) > 1:
                p.service_sync_distance[(min(p.required_services), max(p.required_services))] = tuple(data['delta'][p.index])            
            assert all(map(lambda v: v >=0 , data['delta'][p.index])), "We assume that deltas must be non-negative"
            for s in p.required_services:
                p.processing_times[s] = data['p'][p.index, s]
                # verify that the processing time do not differ among caregivers
                for c in range(self.caregivers - 1):
                    assert np.all(data['p'][p.index, c] == data['p'][p.index, c + 1])
                p.processing_times[s] = data['p'][p.index, 0, s]
            if 'cx' in data and 'cy' in data:
                p.location = (data['cx'][p.index], data['cy'][p.index])
        assert sum(1 for p in self.patients.values() if len(p.required_services) == 2) == data['synch']
        self.travel_distance = np.zeros((len(self.patients) + 1, len(self.patients) + 1))
        coordinates = np.column_stack((data['cx'], data['cy']))
        assert coordinates.shape[0] == len(self.patients) + len(self.depots)
        for i, j in combinations(range(len(self.patients) + 1), 2):
            self.travel_distance[j, i] = self.travel_distance[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])        
        self.depot_location = tuple(coordinates[0])
        self.depot_time_window = (data['e'][min(self.depots)], data['l'][max(self.depots)])
        assert self.depot_time_window[0] == 0.0

    @staticmethod
    def _transform_service(s) -> str:
        if type(s) == str:
            return s
        else:
            return f"s{s + 1}"
        
    @staticmethod
    def _transform_caregiver(c) -> str:
        if type(c) == str:
            return c
        else:
            return f"c{c + 1}"

    def parse_kummer(self, content : str):
        def read_scalar(name: str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid at line {counter}, expecting {pattern}"            
            content.pop(0)
            m = re.match('(\d+)', content[0])
            content.pop(0)
            return counter + 2, conv(m.group(1))
        def read_array(name : str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid: expecting {pattern} at line {counter}"
            if conv is int:
                m = re.match(r'\d+\s+', content[1])
            elif conv is float:
                m = re.match(r'[-+]?(?:\d*\.*\d+)', content[1])
            assert m, f"File format not valid: expecting array {name} at line {counter + 1}"
            res = np.array([conv(v) for v in re.split(r'\s+', content[1]) if v.strip()])
            content.pop(0)
            content.pop(0)            
            return counter + 2, res
        def read_matrix(name : str, content : list[str], counter : int, conv : Callable[[str], Any]):
            pattern = f'{name}'
            m = re.match(pattern, content[0])
            assert m, f"File format not valid: expecting {pattern} at line {counter}"
            content.pop(0)
            l = 1
            res = []
            while True:
                if conv is int:
                    m = re.match(r'\d+\s+', content[0])
                elif conv is float:
                    m = re.match(r'[-+]?(?:\d*\.*\d+)\s+', content[0])
                if not m:
                    break
                res.append([conv(v) for v in re.split(r'\s+', content[0]) if v.strip()])
                l += 1
                content.pop(0)
            return counter + l, np.array(res)

        data = {}
        # this is a very nasty format
        lines = content.split('\n')
        l = 1
        l, data['customers'] = read_scalar('nbNodes', lines, l, int)
        l, data['vehicles'] = read_scalar('nbVehi', lines, l, int)
        l, data['services'] = read_scalar('nbServi', lines, l, int)
        l, data['r'] = read_matrix('r', lines, l, int)
        l, data['double_services'] = read_array('DS', lines, l, int)
        l, data['att'] = read_matrix('a', lines, l, int)
        l, data['cx'] = read_array('x', lines, l, float)
        l, data['cy'] = read_array('y', lines, l, float)
        l, data['d'] = read_matrix('d', lines, l, float)
        l, data['p'] = read_matrix('p', lines, l, float)
        l, data['mind'] = read_array('mind', lines, l, int)
        l, data['maxd'] = read_array('maxd', lines, l, int)
        l, data['e'] = read_array('e', lines, l, int)
        l, data['l'] = read_array('l', lines, l, int)   
        
        self.depots = {0}
        self.patients = {}
        for i in range(data['customers'] - 2):
            patient = Patient(i + 1)
            self.patients[patient.id] = patient
        self.caregivers = data['vehicles']         
        self.service_types = data['services']  
        self.skills = { i: set() for i in range(self.caregivers) }
        assert data['att'].shape[0] == self.caregivers and data['att'].shape[1] == self.service_types
        for c in range(self.caregivers):
            self.skills[c] = set(np.flatnonzero(data['att'][c] > 0))
        self.required_services = {}
        assert data['r'].shape[0] == len(self.patients) + 2 * len(self.depots) 
        assert data['r'].shape[1] == self.service_types        
        assert data['e'].shape[0] == len(self.patients) + 2 * len(self.depots)
        assert data['l'].shape[0] == len(self.patients) + 2 * len(self.depots)
        # assert data['delta'].shape[0] == len(self.patients) + 2 * len(self.depots) and data['delta'].shape[1] == 2
        data['p'] = data['p'].reshape((len(self.patients) + 2 * len(self.depots), self.caregivers, self.service_types))
        assert data['p'].shape[0] == len(self.patients) + 2 * len(self.depots)
        assert data['p'].shape[1] == self.caregivers 
        assert data['p'].shape[2] == self.service_types
        assert data['mind'].shape[0] == len(self.patients) + 2 * len(self.depots)
        assert data['maxd'].shape[0] == len(self.patients) + 2 * len(self.depots)
        if 'cx' in data and 'cy' in data:
            assert data['cx'].shape[0] == len(self.patients) + 2 * len(self.depots) 
            assert data['cy'].shape[0] == len(self.patients) + 2 * len(self.depots)
        for p in self.patients.values():
            p.required_services = set(np.flatnonzero(data['r'][p.index] > 0))
            assert len(p.required_services) <= 2
            p.time_window = (data['e'][p.index], data['l'][p.index])
            if len(p.required_services) > 1:                
                p.service_sync_distance[(min(p.required_services), max(p.required_services))] = (data['mind'][p.index], data['maxd'][p.index])            
            for s in p.required_services:
                # verify that the processing time do not differ among caregivers
                for c in range(self.caregivers - 1):
                    assert np.all(data['p'][p.index, c] == data['p'][p.index, c + 1])
                p.processing_times[s] = data['p'][p.index, 0, s]
            if 'cx' in data and 'cy' in data:
                p.location = (data['cx'][p.index], data['cy'][p.index])
        self.travel_distance = data['d'][:-len(self.depots),:-len(self.depots)]   
        coordinates = np.column_stack((data['cx'], data['cy']))
        assert coordinates.shape[0] == len(self.patients) + 2 * len(self.depots)
        # for i, j in combinations(range(len(self.patients) + 1), 2):        
        #     assert np.abs(self.travel_distance[i, j] - np.linalg.norm(coordinates[i] - coordinates[j])) < 10.0
        #     self.travel_distance[j, i] = self.travel_distance[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])        
        self.depot_location = tuple(coordinates[0])
        self.depot_time_window = (data['e'][min(self.depots)], data['l'][max(self.depots)])
        assert self.depot_time_window[0] == 0.0

    def parse_cplex(self, content):
        # the data file is formatted according to cplex stanard
        equal = Literal('=').suppress()
        identifier = Word(alphas, alphanums + '_').setResultsName("identifier")
        lbrack = Literal("[").suppress()
        rbrack =  Literal("]").suppress()
        lsetbrack = Literal("{").suppress()
        rsetbrack = Literal("}").suppress()
        semi = Literal(";").suppress()
        point = Literal('.')
        comma = Literal(',').suppress()
        e = Literal('E')
        comment = '//' + restOfLine
        plusorminus = Literal('+') | Literal('-')
        digits = Word(nums)
        integer = Combine(Optional(plusorminus) + digits + ~FollowedBy(point)).setParseAction(lambda s, l, t: [int(t[0])])
        floatnumber = Combine(Optional(plusorminus) + digits + point + Optional(digits) + Optional(e + integer)).setParseAction(lambda s, l, t: [float(t[0])])
        number = integer | floatnumber
        array = lbrack + Optional(delimitedList(number, ',', min=1)) + rbrack
        array.setResultsName('array')
        multidim_array = Forward()
        multidim_array <<= array | (lbrack + OneOrMore(multidim_array) + rbrack)
        multidim_array.setParseAction(np.array)
        multidim_array.setResultsName('array')
        iset = lsetbrack + Optional(delimitedList(number, ',')) + rsetbrack
        iset.setResultsName('set')        
        iset.setParseAction(set)
        value = number | multidim_array | iset
        value.setResultsName('value')
        assignment = identifier + equal + value
        grammar = Dict(delimitedList(Group(assignment), ';') + Literal(';').suppress())
        grammar.ignore(comment)
        data = grammar.parseString(content, parseAll=True)

        self.depots = {0, data['nbNodes'] - 1}
        self.patients = {}
        for i in range(data['nbNodes'] - len(self.depots)):
            patient = Patient(i + 1)
            self.patients[patient.id] = patient
        
        self.caregivers = data['nbVehi']
        self.service_types = data['nbServi']
        self.skills = { c: set() for c in range(self.caregivers) }
        assert data['a'].shape[0] == self.caregivers and data['a'].shape[1] == self.service_types
        for c in range(self.caregivers):
            self.skills[c] = set(np.flatnonzero(data['a'][c] > 0))
        self.required_services = {}
        assert data['r'].shape[0] == len(self.patients) + len(self.depots) and data['r'].shape[1] == self.service_types, f"{data['r'].shape} {len(self.patients)} {len(self.depots)}, {self.service_types}"
        assert data['e'].shape[0] == len(self.patients) + len(self.depots)
        assert data['l'].shape[0] == len(self.patients) + len(self.depots)
        assert data['mind'].shape[0] == len(self.patients) + len(self.depots) and data['maxd'].shape[0] == len(self.patients) + len(self.depots)
        assert data['p'].shape[0] == len(self.patients) + len(self.depots) and data['p'].shape[1] == self.caregivers and data['p'].shape[2] == self.service_types
        data['DS'] = set(data['DS'])
        for p in self.patients.values():
            p.required_services = sorted(np.flatnonzero(data['r'][p.index] > 0))
            assert len(p.required_services) <= 2 and (len(p.required_services) == 1 or p.index + 1 in data['DS']), f"{p} has either more than 2 services or the patient {p.index} is not appearing in double services {data['DS']}"
            p.time_window = (data['e'][p.index], data['l'][p.index])
            if len(p.required_services) > 1:
                p.service_sync_distance[(p.required_services[0], p.required_services[1])] = (data['mind'][p.index], data['maxd'][p.index])
            for s in p.required_services:
                # verify that the processing time do not differ among caregivers
                for c in range(self.caregivers - 1):
                    assert np.all(data['p'][p.index, c] == data['p'][p.index, c + 1])
                p.processing_times[s] = data['p'][p.index, 0, s]
            if 'x' in data and 'y' in data:
                p.location = (data['x'][p.index], data['y'][p.index])
#        self.travel_distance = np.zeros((len(self.patients) + 1, len(self.patients) + 1))
        self.travel_distance = data['d'][:-(len(self.depots) - 1),:-(len(self.depots) - 1)]
        assert self.travel_distance.shape[0] == len(self.patients) + 1, f"{self.travel_distance.shape}"
        self.depot_location = (data['x'][0], data['y'][0])
        self.depot_time_window = (data['e'][min(self.depots)], data['l'][max(self.depots)])
        assert self.depot_time_window[0] == 0.0

    def dispatch_json(self, json_instance : dict):
        self.depots = {0}
        self.depot_location = tuple(json_instance['central_offices'][0]['location']) if 'location' in json_instance['central_offices'][0] else None
        # FIXME: currently not set
        self.depot_time_window = (0, float('inf'))
        self.caregivers = len(json_instance['caregivers'])         
        self.service_types = len(json_instance['services'])
        self.skills = {c['id']: set() for c in json_instance['caregivers']}
        self.services = [s['id'] for s in json_instance['services']]
        self.patients = { p['id']: Patient(i + 1, p['id']) for i, p in enumerate(json_instance['patients']) }
        for c in json_instance['caregivers']:
            self.skills[c['id']] = set(c['abilities'])
        self.required_services = {}
        for i, p in enumerate(self.patients.values()):
            p.required_services = [c['service'] for c in json_instance['patients'][i]['required_caregivers']]
            p.processing_times = { c['service'] : c['duration'] for c in json_instance['patients'][i]['required_caregivers'] }
            assert len(p.required_services) <= 2
            p.time_window = json_instance['patients'][i]['time_window']
            p.location = json_instance['patients'][i]['location'] if 'location' in json_instance['patients'][i] else None
            if len(p.required_services) > 1:                
                if json_instance['patients'][i]['synchronization']['type'] == "sequential":
                    p.service_sync_distance[p.required_services[0], p.required_services[1]] = tuple(json_instance['patients'][i]['synchronization']['distance'])
                else:
                    p.service_sync_distance[p.required_services[0], p.required_services[1]] = (0, 0)

        self.travel_distance = np.array(json_instance['distances'])    

    def to_json(self) -> dict:
        result = dict()
        result['patients'] = list()
        for p in self.patients.values():
            patient = { 'id': p.id, 'location': list(map(float, p.location)),  'time_window': list(map(float, p.time_window)), 'required_caregivers': [] }              
            for s in p.required_services: 
                patient['required_caregivers'].append({'service': self._transform_service(s), 'duration': float(p.processing_times[s]) })
            if len(p.required_services) > 1:
                patient['synchronization'] = dict()
                # TODO: currently it relies on the fact that at most two services are required
                for dist in p.service_sync_distance.values():                    
                    if tuple(dist) == (0, 0):
                        patient['synchronization']['type'] = 'simultaneous'
                    else:                    
                        patient['synchronization']['type'] = 'sequential'
                        patient['synchronization']['distance'] = list(map(int, dist))
                    break
            result['patients'].append(patient) 
        if not hasattr(self, 'services'):
            self.services = [self._transform_service(s) for s in range(self.service_types)]
        result['services'] = list()                        
        durations = defaultdict(list)
        for p in self.patients.values():
            for s in p.required_services:
                durations[self._transform_service(s)].append(p.processing_times[s])  
        default_duration = defaultdict(lambda: 30)
        for s, d in durations.items():
            m = st.mode(d, keepdims=False)
            default_duration[self._transform_service(s)] = m.mode
        for s in self.services:
            result['services'].append({'id': s, 'default_duration': float(default_duration[s]) })
        result['caregivers'] = list()
        for c, skills in self.skills.items():
            caregiver = { 'id': self._transform_caregiver(c), 'abilities': list(map(self._transform_service, skills)) }     
            result['caregivers'].append(caregiver)
        result['central_offices'] = [{ 'id': 'd', 'location': list(map(float, self.depot_location)) }]
        result['distances'] = self.travel_distance.tolist()
        return result

    def __repr__(self) -> str:
        return f"Patients = {len(self.patients)}, Caregivers = {self.caregivers}, Service Types = {self.service_types}\nSkills = {self.skills}\n"

    def compute_features(self):
        features = {}
        features['patients'] = { 'total': len(self.patients), 'single': sum(len(p.required_services) == 1 for p in self.patients.values()) / len(self.patients), 'double': sum(len(p.required_services) == 2 for p in self.patients.values()) / len(self.patients), 'simultaneous': sum(len(p.required_services) == 2 and p.service_sync_distance == (0, 0) for p in self.patients.values()), 'sequential': sum(p.service_sync_distance != (0, 0) for p in self.patients.values() for p in self.patients.values()) }
        features['caregivers'] = self.caregivers
        features['service_types'] = self.service_types
        compatible_caregivers= []
        compatible_multiskill_caregivers = []
        patients_with_compatible_multiskill_caregivers = set()
        time_windows_size = []
        service_length = []
        for p in self.patients.values():
            for s in p.required_services:
                compatible_caregivers.append(sum(s in skills for skills in self.skills.values()) / self.caregivers)
            tmp_cmc = 0
            if len(p.required_services) > 1:
                if type(p.required_services) is set:
                    sync_distance = p.service_sync_distance[(min(p.required_services), max(p.required_services))]
                else:
                    sync_distance = p.service_sync_distance[(p.required_services[0], p.required_services[1])]
                if max(p.processing_times[min(p.required_services)], sync_distance[0]) <= sync_distance[1]: # can be performed by the same caregiver
                    for skills in self.skills.values():
                        if skills >= set(p.required_services):
                            tmp_cmc += 1
                            patients_with_compatible_multiskill_caregivers.add(p)                    
            compatible_multiskill_caregivers.append(tmp_cmc / self.caregivers)
            time_windows_size.append(float(p.time_window[1] - p.time_window[0]))
            service_length += p.processing_times.values()
        tmp = list(map(lambda s: len(s) / self.service_types, self.skills.values()))
        features['ability_rate'] = { 'min': min(tmp), 'avg': sum(tmp) / len(tmp), 'max': max(tmp) }
        features['compatible_caregivers_rate'] = { 'min': min(compatible_caregivers), 'avg': sum(compatible_caregivers) / len(compatible_caregivers), 'max': max(compatible_caregivers) }
        features['patients_with_compatible_multiskill_caregivers'] = len(patients_with_compatible_multiskill_caregivers)
        features['compatible_multiskill_caregivers'] = { 'min': min(compatible_multiskill_caregivers), 'avg': sum(compatible_multiskill_caregivers) / len(compatible_multiskill_caregivers), 'max': max(compatible_multiskill_caregivers) }
        features['time_windows_size'] = { 'min': min(time_windows_size), 'avg': sum(time_windows_size) / len(self.patients), 'max': max(time_windows_size) }
        features['service_length'] = { 'min': min(service_length), 'avg': sum(service_length) / len(service_length), 'max': max(service_length) }
        mask = np.ones(self.travel_distance.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        features['distances'] = { 'min': self.travel_distance[mask].min(), 'avg': self.travel_distance[mask].mean(), 'max': self.travel_distance[mask].max() }
    
        return features

    def to_graph(self, scale : float = 0.5):
        if graphviz_available:
            g = graphviz.Graph(comment='Home Healthcare RSP')
            g.graph_attr['layout'] = 'neato'
            g.node('0', 'd_0',  pos=f'{self.depot_location[0] * scale},{self.depot_location[1] * scale}!', color='lightgray', style='filled', shape='rect')
            for p in self.patients.values():
                g.node(f'{p.index}', f'{p.index}', pos=f'{p.location[0] * scale},{p.location[1] * scale}!')
            for i, j in combinations(range(len(self.patients) + 1), 2):
                g.edge(f'{i}', f'{j}', label=f'{self.travel_distance[i][j]:.1f}')
            g.view()
