import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os.path as osp
import pandas as pd
import quaternion



init_pos = (16,19)
heading = 270

# ===== 1. 벽 정보 가져오기 ==================
data_walls = pd.read_csv(osp.join('./data', 'walls2.csv'))
grid= np.array(data_walls[['width','height']])
grid_size = grid[0]
walls = []
walls = np.array(data_walls[['x','y']]).tolist()
walls = [tuple(wall) for wall in walls]

valid_states = [(x, y) for x in range(0,grid_size[0]) for y in range(0,grid_size[1]) if (x, y) not in tuple(walls)]
double_states = [(x, y) for x in range(0,grid_size[0],2) for y in range(0,grid_size[1],2) if (x, y) not in tuple(walls)]
# ===== 2. 관측값 (상대 이동 벡터) =====
df = pd.read_csv(osp.join('./data', 'dataPF.csv'))

# magnetic 함수 
mag = np.array(df[['mag-x','mag-y','mag-z']])
ori = np.array(df[['qw','qx','qy','qz']])
mag_q = quaternion.from_float_array(np.concatenate([np.zeros([mag.shape[0], 1]), mag],axis = 1))
ori_q = quaternion.from_float_array(ori)
glob_mag = quaternion.as_float_array(ori_q*mag_q*ori_q.conj())[:,1:]

dmagnetic = []

#PDR Trajectory
traj = np.array(df[['dX','dY']])
traj = traj*np.array([2,-2])
traj_sum = np.cumsum(traj,axis=0)

interval = 1
relative_traj = []
target_distance = 0

diff = np.linalg.norm(np.diff(traj_sum,axis = 0),axis=1)
cum_distance = np.cumsum(diff)
sample_num = 0

# 1m 마다 samping
for i,distance in enumerate(cum_distance):
    if distance >= target_distance:
        relative_traj.append(traj_sum[i])

        mag_average = np.average([glob_mag[i] for i in range(int(i-(i-sample_num)*0.3),i+1)],axis=0)
        dmagnetic.append(mag_average)
        target_distance += 2
        sample_num = i

# ===== 3. 회전 및 슬라이딩 함수 =====
def rotate_trajectory(traj, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    traj_np = np.array(traj)
    # center = np.mean(traj_np, axis=0)
    # rotated = np.dot(traj_np - center, rot_matrix.T) + center
    rotated = np.dot(traj_np, rot_matrix.T)
    return [rotated]

def slide_trajectory(traj, dx, dy):
    return [(x + dx, y + dy) for (x, y) in traj]

# ===== 4. HMM  =====

def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def emission_prob(obs_pos, state, sigma=0.5,alpha=0.5):
    d_pos = np.linalg.norm(np.array(obs_pos) - np.array(state))
    p_pos = np.exp(-d_pos**2 / (2 * sigma**2))
    return p_pos 

def transition_prob(s1, s2):
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    interwall = (int((s2[0]+s1[0])/2),int((s2[1]+s1[1])/2))

        
    if abs(dx) <= 2 and abs(dy) <= 2 and s2 in valid_states and interwall not in tuple(walls) :
        prob = 1.0 

    else:
        prob = 0.0

    return prob  

def get_local_states(center, all_states, radius=6.0):   # 거리 6m 의 좌표만 
    return [s for s in all_states if np.linalg.norm(np.array(s) - np.array(center)) <= radius]


# ===================== 5. 전체 탐색 (회전 + 슬라이딩 + HMM) ===========================
angles = list(range(0,360,10))

delta_list = []
path = []
local_states_list = []
scores = []
radius = 6.0

# 초기 상태 후보 필터링
local_states = get_local_states(init_pos, double_states, radius)
local_states_list.append(local_states)

# 초기 delta, psi
delta = {s: np.log(emission_prob(init_pos, s) + 1e-9) for s in local_states}
psi = {s: None for s in local_states}

for i in range(len(angles)):
    delta_list.append(delta) 
    path.append([])




for traj in relative_traj:
    delta_rot=[]
    top3_candidate = []
    maxstate_rot = {}
    for r, ang in enumerate(angles):
        rotated = rotate_trajectory(traj, ang)
        obs = slide_trajectory(rotated, init_pos[0], init_pos[1])

        delta_t = {}
        psi_t = {}
        prev_states = local_states_list[-1]
        local_states = get_local_states(obs, double_states, radius)
        local_states_list.append(local_states)

        for s_j in local_states:
            max_val = -np.inf
            max_state = None
            
            for s_i in prev_states:
                prev_delta = delta_list[r].get(s_i, -np.inf) #s_i 의 값이 없으면 -np.inf 반환
                if prev_delta == -np.inf:
                    continue  # 이전 스텝에서 유효하지 않음

                tr = transition_prob(s_i, s_j)
                if tr == 0:
                    continue  # 연결 안 됨

                val = prev_delta + np.log(tr + 1e-9) # log0 = -inf, log1 = 0:
                
                if val > max_val:
                    max_val = val
                    max_state = s_i

            if max_state is not None:
                delta_t[s_j] = max_val + np.log(emission_prob(obs, s_j) + 1e-9)
                psi_t[s_j] = max_state

        delta_list[r] = delta_t
        if not delta_list[r]:
            continue
        maxstate = max(delta_list[r], key=delta_list[r].get)
        top3_candidate.append({maxstate : [r,delta_list[r][maxstate]]})
        path[r].append(maxstate)
    top3 = sorted(top3_candidate, key=lambda x: list(x.values())[0][1], reverse=True)[:3]
    print(top3)
      
        
