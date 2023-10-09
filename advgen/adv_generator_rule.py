import numpy as np
from collections import deque
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import tensorflow as tf
import advgen.utils, advgen.structs, advgen.globals
from advgen.modeling.vectornet import VectorNet
from advgen.adv_utils import process_data
import bezier

MDAgentTypeConvert = dict(
    VEHICLE = 1,
    PEDESTRIAN = 2,
    CYCLIST = 3,
    OTHERS = 4,
)

MDMapTypeConvert = dict(
    LANE_FREEWAY = 1,
    LANE_SURFACE_STREET = 2,
    LANE_BIKE_LANE = 3,
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6,
    ROAD_LINE_SOLID_SINGLE_WHITE = 7,
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8,
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9,
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10,
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11,
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12,
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13,
    ROAD_EDGE_BOUNDARY = 15,
    ROAD_EDGE_MEDIAN = 16,
    STOP_SIGN = 17,
    CROSSWALK = 18,
    SPEED_BUMP = 19,
)

MDLightTypeConvert = dict(
    LANE_STATE_UNKNOWN = 0,
    LANE_STATE_ARROW_STOP = 1,
    LANE_STATE_ARROW_CAUTION = 2,
    LANE_STATE_ARROW_GO = 3,
    LANE_STATE_STOP = 4,
    LANE_STATE_CAUTION = 5,
    LANE_STATE_GO = 6,
    LANE_STATE_FLASHING_STOP = 7,
    LANE_STATE_FLASHING_CAUTION = 8,
)

def moving_average(data, window_size):
    interval = np.pad(data,window_size//2,'edge')
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(interval, window, 'valid')
    return res


def get_polyline_dir(polyline):
    if polyline.ndim == 1:
        return np.zeros(3)
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

def get_polyline_yaw(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    diff = polyline_post - polyline
    polyline_yaw = np.arctan2(diff[:,1],diff[:,0])
    polyline_yaw[-1] = polyline_yaw[-2]
    #polyline_yaw = np.where(polyline_yaw<0,polyline_yaw+2*np.pi,polyline_yaw)
    for i in range(len(polyline_yaw)-1):
        if polyline_yaw[i+1] - polyline_yaw[i] > 1.5*np.pi:
            polyline_yaw[i+1] -= 2*np.pi
        elif polyline_yaw[i] - polyline_yaw[i+1] > 1.5*np.pi:
            polyline_yaw[i+1] += 2*np.pi
    return moving_average(polyline_yaw, window_size = 5)

def get_polyline_vel(polyline):
    polyline_post = np.roll(polyline, shift=-1, axis=0)
    polyline_post[-1] = polyline[-1]
    diff = polyline_post - polyline
    polyline_vel = diff / 0.1
    return polyline_vel

###   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
def Intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False


class AdvGenerator():
    def __init__(self,parser):
        advgen.utils.add_argument(parser)
        parser.set_defaults(other_params=['l1_loss','densetnt', 'goals_2D', 'enhance_global_graph' ,'laneGCN' ,'point_sub_graph', 'laneGCN-4' ,'stride_10_2' ,'raster' ,'train_pair_interest'])
        parser.set_defaults(mode_num=32)
        args = parser.parse_args()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
        logger = logging.getLogger(__name__)    
        advgen.utils.init(args,logger)

        self.model = VectorNet(args).to(0)
        self.model.eval()

        self.model.load_state_dict(torch.load('./advgen/pretrained/densetnt.bin'))

        self.args = args

        self.storage = {}
    
    def before_episode(self,env):
        self.env = env
        self.ego_traj = []
        self.adv_traj = []
        self.adv_name = None
        if not self.storage.get(self.env.current_seed):
            traffic_motion_feat,adv_agent,ego_navigation_route,adv_past,adv_navigation_route = self._parse()
            
            AV_trajs = deque(maxlen=self.args.AV_traj_num)

            assert self.args.AV_traj_num == 1
            
            for _ in range(self.args.AV_traj_num):
                AV_trajs.append(ego_navigation_route)

            AV_trajs_eval = deque(maxlen=1)
            AV_trajs_eval.append(ego_navigation_route)
            
            ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
            try:
                adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
            except:
                adv_obj = ego_obj
            self.storage[self.env.current_seed] = dict(
                traffic_motion_feat = traffic_motion_feat,
                adv_agent = adv_agent,
                adv_past = adv_past,
                adv_navigation_route = adv_navigation_route,
                adv_info = dict(w=adv_obj.top_down_width,l=adv_obj.top_down_length),
                ego_info = dict(w=ego_obj.top_down_width,l=ego_obj.top_down_length),
                AV_trajs = AV_trajs,
                AV_trajs_eval = AV_trajs_eval,
            )
        
        self.ego_vel = []
        self.ego_heading = []
        
    
    def after_episode(self,update_AV_traj=False,mode='train'):
        if update_AV_traj:
            latest_ego_traj = np.array(self.ego_traj)[11:91]
            if len(latest_ego_traj)<10:
                print('Ignore traj less than 1s')
                return
            if mode == 'train':
                self.storage[self.env.current_seed]['AV_trajs'].append(latest_ego_traj)
            elif mode == 'eval':
                self.storage[self.env.current_seed]['AV_trajs_eval'].append(latest_ego_traj)
            else:
                raise NotImplementedError

    

    def log_AV_history(self):
        obj = self.env.engine.get_object('default_agent').get('default_agent')
        self.ego_traj.append(obj.position)
        self.ego_vel.append(obj.velocity)
        self.ego_heading.append(obj.heading_theta)


    def _parse(self):
        scenario_data = self.env.engine.data_manager._scenario[self.env.current_seed]
        
        default_agent = scenario_data['metadata']['sdc_id']
        objects_of_interest = scenario_data['metadata']['objects_of_interest']
        assert len(objects_of_interest) == 2 and default_agent in objects_of_interest
        objects_of_interest.remove(default_agent)
        adv_agent = objects_of_interest[0]
        
        raw_map_features = scenario_data['map_features']
        raw_dynamic_map_states = scenario_data['dynamic_map_states']
        raw_tracks_features = scenario_data['tracks']

        tracks_ids = list(raw_tracks_features.keys())
        tracks_ids.remove(default_agent)
        tracks_ids.remove(adv_agent)
        tracks_ids = [default_agent,adv_agent] + tracks_ids
       
        map_features = {
        'roadgraph_samples/dir': np.full([20000,3], -1 , dtype=np.float32),
        'roadgraph_samples/id': np.full([20000,1], -1 , dtype=np.int64),
        'roadgraph_samples/type': np.full([20000,1], -1 , dtype=np.int64),
        'roadgraph_samples/valid': np.full([20000,1], 1 , dtype=np.int64),
        'roadgraph_samples/xyz': np.full([20000,3], -1 , dtype=np.float32),
        }

        state_features = {
            'state/id': np.full([128,], -1 , dtype=np.int64),
            'state/type': np.full([128,], 0 , dtype=np.int64),
            'state/is_sdc': np.full([128,], 0 , dtype=np.int64),
            'state/tracks_to_predict': np.full([128,], 0 , dtype=np.int64),
            'state/current/bbox_yaw': np.full([128,1],-1 , dtype=np.float32),
            'state/current/height': np.full([128,1], -1 , dtype=np.float32),
            'state/current/length': np.full([128,1], -1 , dtype=np.float32),
            'state/current/valid':np.full([128,1], 0 , dtype=np.int64),
            'state/current/vel_yaw':np.full([128,1], -1 , dtype=np.float32),
            'state/current/velocity_x': np.full([128,1], -1 , dtype=np.float32),
            'state/current/velocity_y': np.full([128,1], -1 , dtype=np.float32),
            'state/current/width': np.full([128,1], -1 , dtype=np.float32),
            'state/current/x': np.full([128,1], -1 , dtype=np.float32),
            'state/current/y': np.full([128,1], -1 , dtype=np.float32),
            'state/current/z': np.full([128,1], -1 , dtype=np.float32),
            'state/past/bbox_yaw': np.full([128,10], -1 , dtype=np.float32),
            'state/past/height': np.full([128,10], -1 , dtype=np.float32),
            'state/past/length': np.full([128,10], -1 , dtype=np.float32),
            'state/past/valid':np.full([128,10], 0 , dtype=np.int64),
            'state/past/vel_yaw':np.full([128,10], -1 , dtype=np.float32),
            'state/past/velocity_x': np.full([128,10], -1 , dtype=np.float32),
            'state/past/velocity_y': np.full([128,10], -1 , dtype=np.float32),
            'state/past/width': np.full([128,10], -1 , dtype=np.float32),
            'state/past/x': np.full([128,10], -1 , dtype=np.float32),
            'state/past/y': np.full([128,10], -1 , dtype=np.float32),
            'state/past/z': np.full([128,10], -1 , dtype=np.float32),
            'state/future/bbox_yaw': np.full([128,80], -1 , dtype=np.float32),
            'state/future/height': np.full([128,80], -1 , dtype=np.float32),
            'state/future/length': np.full([128,80], -1 , dtype=np.float32),
            'state/future/valid':np.full([128,80], 0 , dtype=np.int64),
            'state/future/vel_yaw':np.full([128,80], -1 , dtype=np.float32),
            'state/future/velocity_x': np.full([128,80], -1 , dtype=np.float32),
            'state/future/velocity_y': np.full([128,80], -1 , dtype=np.float32),
            'state/future/width': np.full([128,80], -1 , dtype=np.float32),
            'state/future/x': np.full([128,80], -1 , dtype=np.float32),
            'state/future/y': np.full([128,80], -1 , dtype=np.float32),
            'state/future/z': np.full([128,80], -1 , dtype=np.float32),

        }

        traffic_light_features = {
            'traffic_light_state/current/state': np.full([1,16], -1 , dtype=np.int64),
            'traffic_light_state/current/valid': np.full([1,16], 0 , dtype=np.int64),
            'traffic_light_state/current/id': np.full([1,16], -1 , dtype=np.int64),
            'traffic_light_state/current/x': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/current/y': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/current/z': np.full([1,16], -1 , dtype=np.float32),
            'traffic_light_state/past/state': np.full([10,16], -1 , dtype=np.int64),
            'traffic_light_state/past/valid': np.full([10,16], 0 , dtype=np.int64),
            'traffic_light_state/past/x': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/y': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/z': np.full([10,16], -1 , dtype=np.float32),
            'traffic_light_state/past/id': np.full([10,16], -1 , dtype=np.int64),
        }

        count = 0

        for k,v in raw_map_features.items():
            _id = int(k)
            _type = MDMapTypeConvert[v['type']]

            if _type in [17]:
                _poly = v['position']
            elif _type in [18,19]:
                _poly = v['polygon']
            else:
                _poly = v['polyline']
            
            _dir = get_polyline_dir(_poly)

            
            # clip > 20000
            try:
                map_features['roadgraph_samples/xyz'][count:count+len(_poly)] = _poly
                map_features['roadgraph_samples/dir'][count:count+len(_poly)] = _dir
                map_features['roadgraph_samples/id'][count:count+len(_poly)] = _id
                map_features['roadgraph_samples/type'][count:count+len(_poly)] = _type
            except:
                map_features['roadgraph_samples/xyz'][count:20000] = _poly[:20000-count]
                map_features['roadgraph_samples/dir'][count:20000] = _dir[:20000-count]
                map_features['roadgraph_samples/id'][count:20000] = _id
                map_features['roadgraph_samples/type'][count:20000] = _type
                break

            count += len(_poly)


        tracks_ids = tracks_ids[:128]

        state_features['state/id'][:len(tracks_ids)] = tracks_ids
        state_features['state/is_sdc'][0] = 1
        state_features['state/tracks_to_predict'][:2] = 1

        for i, track_id in enumerate(tracks_ids):

            track_data = raw_tracks_features.get(track_id)
            
            # construct ego navigation route
            if i == 0:
                ego_navigation_route = track_data['state']['position'][11:,:2]
            
            if i == 1:
                adv_past = track_data['state']['position'][:11,:2]
                adv_navigation_route = track_data['state']['position'][11:,:2]

            state_features['state/type'][i] = MDAgentTypeConvert[track_data['type']]

            for j in range(0,10):
                state_features['state/past/x'][i][j] = track_data['state']['position'][j][0]
                state_features['state/past/y'][i][j] = track_data['state']['position'][j][1]
                state_features['state/past/z'][i][j] = track_data['state']['position'][j][2]
                state_features['state/past/bbox_yaw'][i][j] = track_data['state']['heading'][j]
                state_features['state/past/velocity_x'][i][j] = track_data['state']['velocity'][j][0]
                state_features['state/past/velocity_y'][i][j] = track_data['state']['velocity'][j][1]
                state_features['state/past/vel_yaw'][i][j] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/past/width'][i][j] = track_data['state']['width'][j]
                state_features['state/past/height'][i][j] = track_data['state']['height'][j]
                state_features['state/past/length'][i][j] = track_data['state']['length'][j]
                state_features['state/past/valid'][i][j] = track_data['state']['valid'][j]
            

            for j in range(10,11):
                state_features['state/current/x'][i] = track_data['state']['position'][j][0]
                state_features['state/current/y'][i] = track_data['state']['position'][j][1]
                state_features['state/current/z'][i] = track_data['state']['position'][j][2]
                state_features['state/current/bbox_yaw'][i] = track_data['state']['heading'][j]
                state_features['state/current/velocity_x'][i] = track_data['state']['velocity'][j][0]
                state_features['state/current/velocity_y'][i] = track_data['state']['velocity'][j][1]
                state_features['state/current/vel_yaw'][i] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/current/width'][i] = track_data['state']['width'][j]
                state_features['state/current/height'][i] = track_data['state']['height'][j]
                state_features['state/current/length'][i] = track_data['state']['length'][j]
                state_features['state/current/valid'][i] = track_data['state']['valid'][j]

            
            for j in range(11,91):
                state_features['state/future/x'][i][j-11] = track_data['state']['position'][j][0]
                state_features['state/future/y'][i][j-11] = track_data['state']['position'][j][1]
                state_features['state/future/z'][i][j-11] = track_data['state']['position'][j][2]
                state_features['state/future/bbox_yaw'][i][j-11] = track_data['state']['heading'][j]
                state_features['state/future/velocity_x'][i][j-11] = track_data['state']['velocity'][j][0]
                state_features['state/future/velocity_y'][i][j-11] = track_data['state']['velocity'][j][1]
                state_features['state/future/vel_yaw'][i][j-11] = np.arctan2(track_data['state']['velocity'][j][1],track_data['state']['velocity'][j][0])
                state_features['state/future/width'][i][j-11] = track_data['state']['width'][j]
                state_features['state/future/height'][i][j-11] = track_data['state']['height'][j]
                state_features['state/future/length'][i][j-11] = track_data['state']['length'][j]
                state_features['state/future/valid'][i][j-11] = track_data['state']['valid'][j]
        
        for i,v in enumerate(raw_dynamic_map_states.values()):
            if i == 16: break

            if v['type'] != 'TRAFFIC_LIGHT': continue

            for j in range(0,10):
                _state = v['state']['object_state'][j]
                if _state: 
                    traffic_light_features['traffic_light_state/past/state'][j][i] = MDLightTypeConvert[_state]
                    traffic_light_features['traffic_light_state/past/valid'][j][i] = 1
                    traffic_light_features['traffic_light_state/past/id'][j][i] = int(v['lane'])
                    traffic_light_features['traffic_light_state/past/x'][j][i] = v['stop_point'][0]
                    traffic_light_features['traffic_light_state/past/y'][j][i] = v['stop_point'][1]
                    traffic_light_features['traffic_light_state/past/z'][j][i] = v['stop_point'][2]
            
            _state = v['state']['object_state'][10]
            if _state: 
                traffic_light_features['traffic_light_state/current/state'][0][i] = MDLightTypeConvert[_state]
                traffic_light_features['traffic_light_state/current/valid'][0][i] = 1
                traffic_light_features['traffic_light_state/current/id'][0][i] = int(v['lane'])
                traffic_light_features['traffic_light_state/current/x'][0][i] = v['stop_point'][0]
                traffic_light_features['traffic_light_state/current/y'][0][i] = v['stop_point'][1]
                traffic_light_features['traffic_light_state/current/z'][0][i] = v['stop_point'][2]
        
        features_description = {}
        features_description.update(map_features)
        features_description.update(state_features)
        features_description.update(traffic_light_features)
        features_description['scenario/id'] = np.array(['template'])
        features_description['state/objects_of_interest'] = state_features['state/tracks_to_predict'].copy()
        for k,v in features_description.items():
            features_description[k] = tf.convert_to_tensor(v)

        return features_description,adv_agent,ego_navigation_route,adv_past,adv_navigation_route

    @property
    def adv_agent(self):
        return self.storage[self.env.current_seed].get('adv_agent')
    
    def generate(self,mode='train'):
        traffic_motion_feat = self.storage[self.env.current_seed].get('traffic_motion_feat')
        if mode == 'train':
            trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs'))[0]
            trajs_OV = np.array(self.storage[self.env.current_seed].get('adv_navigation_route'))
            
            _AV_len = int(len(trajs_AV)*0.75)
            a = trajs_OV[0]
            b = trajs_OV[int(_AV_len/3)]
            c = trajs_AV[int(_AV_len*2/3)]
            d = trajs_AV[_AV_len]
            points = np.array([[a[0],b[0],c[0],d[0]], [a[1],b[1],c[1],d[1]]])
            curve = bezier.Curve(points, degree=3)
            s_vals = np.linspace(0.0, 1.0, _AV_len)
            res = curve.evaluate_multi(s_vals).transpose((1,0))
            
            adv_past = self.storage[self.env.current_seed].get('adv_past')
            adv_pos = np.concatenate((adv_past,res,trajs_AV[_AV_len+1:]),axis=0)
            adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
            adv_vel = get_polyline_vel(adv_pos)
            
            self.adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))


            return traffic_motion_feat,self.adv_traj,trajs_AV,True


        elif mode == 'eval':
            trajs_AV = np.array(self.storage[self.env.current_seed].get('AV_trajs_eval'))
            probs_AV = [1.]

            batch_data = process_data(traffic_motion_feat,self.args)
            pred_trajectory, pred_score, _ = self.model(batch_data[0], 'cuda')


            trajs_OV = pred_trajectory[1]
            probs_OV = pred_score[1]
            probs_OV[6:] = probs_OV[6]
            probs_OV = np.exp(probs_OV)
            probs_OV = probs_OV / np.sum(probs_OV)


            res = np.zeros(32)
            min_dist = np.full(32,fill_value=1000000)

            for j,prob_OV in enumerate(probs_OV):
                P1 = prob_OV
                traj_OV = trajs_OV[j][::5]
                yaw_OV = get_polyline_yaw(trajs_OV[j])[::5].reshape(-1,1)
                width_OV = self.storage[self.env.current_seed]['adv_info']['w']
                length_OV = self.storage[self.env.current_seed]['adv_info']['l']
                cos_theta = np.cos(yaw_OV)
                sin_theta = np.sin(yaw_OV)
                bbox_OV = np.concatenate((traj_OV,yaw_OV,\
                        traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                        traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta-0.5*width_OV*cos_theta,\
                        traj_OV[:,0].reshape(-1,1)+0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                        traj_OV[:,1].reshape(-1,1)+0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                        traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta-0.5*width_OV*sin_theta,\
                        traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta+0.5*width_OV*cos_theta,\
                        traj_OV[:,0].reshape(-1,1)-0.5*length_OV*cos_theta+0.5*width_OV*sin_theta,\
                        traj_OV[:,1].reshape(-1,1)-0.5*length_OV*sin_theta-0.5*width_OV*cos_theta),axis=1)


                for i,prob_AV in enumerate(probs_AV):
                    P2 = prob_AV
                    traj_AV = trajs_AV[i][::5]
                    yaw_AV = get_polyline_yaw(trajs_AV[i])[::5].reshape(-1,1)
                    width_AV = self.storage[self.env.current_seed]['ego_info']['w']
                    length_AV = self.storage[self.env.current_seed]['ego_info']['l']
                    cos_theta = np.cos(yaw_AV)
                    sin_theta = np.sin(yaw_AV)
                    

                    bbox_AV = np.concatenate((traj_AV,yaw_AV,\
                        traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                        traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta-0.5*width_AV*cos_theta,\
                        traj_AV[:,0].reshape(-1,1)+0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                        traj_AV[:,1].reshape(-1,1)+0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                        traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta-0.5*width_AV*sin_theta,\
                        traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta+0.5*width_AV*cos_theta,\
                        traj_AV[:,0].reshape(-1,1)-0.5*length_AV*cos_theta+0.5*width_AV*sin_theta,\
                        traj_AV[:,1].reshape(-1,1)-0.5*length_AV*sin_theta-0.5*width_AV*cos_theta),axis=1)

                    
                    P3 = 0
                    '''
                    B-A  F-E
                    | |  | |
                    C-D  G-H
                    
                    '''
                    #step = 0
                    for (Cx1,Cy1,yaw1,xA,yA,xB,yB,xC,yC,xD,yD),(Cx2,Cy2,yaw2,xE,yE,xF,yF,xG,yG,xH,yH) in zip(bbox_AV,bbox_OV):
                        ego_adv_dist = np.linalg.norm([Cx1-Cx2,Cy1-Cy2])
                        if ego_adv_dist < min_dist[j]:
                            min_dist[j] = ego_adv_dist
                        if ego_adv_dist >= np.linalg.norm([0.5*length_AV,0.5*width_AV]) + np.linalg.norm([0.5*length_OV,0.5*width_OV]):
                            pass
                        elif Intersect([xA,yA,xB,yB],[xE,yE,xF,yF]) or Intersect([xA,yA,xB,yB],[xF,yF,xG,yG]) or\
                            Intersect([xA,yA,xB,yB],[xG,yG,xH,yH]) or Intersect([xA,yA,xB,yB],[xH,yH,xE,yE]) or\
                            Intersect([xB,yB,xC,yC],[xE,yE,xF,yF]) or Intersect([xB,yB,xC,yC],[xF,yF,xG,yG]) or\
                            Intersect([xB,yB,xC,yC],[xG,yG,xH,yH]) or Intersect([xB,yB,xC,yC],[xH,yH,xE,yE]) or\
                            Intersect([xC,yC,xD,yD],[xE,yE,xF,yF]) or Intersect([xC,yC,xD,yD],[xF,yF,xG,yG]) or\
                            Intersect([xC,yC,xD,yD],[xG,yG,xH,yH]) or Intersect([xC,yC,xD,yD],[xH,yH,xE,yE]) or\
                            Intersect([xD,yD,xA,yA],[xE,yE,xF,yF]) or Intersect([xD,yD,xA,yA],[xF,yF,xG,yG]) or\
                            Intersect([xD,yD,xA,yA],[xG,yG,xH,yH]) or Intersect([xD,yD,xA,yA],[xH,yH,xE,yE]):
                            P3 = 1

                            break

                    res[j] += P1*P2*P3


            if np.any(res):
                adv_traj_id = np.argmax(res)
            else:
                adv_traj_id = np.argmin(min_dist)
            adv_future = trajs_OV[adv_traj_id]
            adv_past = self.storage[self.env.current_seed].get('adv_past')
            adv_pos = np.concatenate((adv_past,adv_future),axis=0)
            adv_yaw = get_polyline_yaw(adv_pos).reshape(-1,1)
            adv_vel = get_polyline_vel(adv_pos)
            self.adv_traj = list(np.concatenate((adv_pos,adv_vel,adv_yaw),axis=1))

            return traffic_motion_feat,self.adv_traj,trajs_AV,any(res)
        