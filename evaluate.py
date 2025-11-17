import numpy as np
import argparse
from common.h36m_dataset import Human36mDataset
from common.data_utils import split_data
from common.data_utils import normalation
from common.data_utils import HumanDataset
# from common.model import Net
from common.IIVFormer import IIVFormer
from tqdm import tqdm
from common.loss import mpjpe
from common.loss import p_mpjpe
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_3d', type=str, default='data/data_3d_h36m.npz')
    # parser.add_argument('--data_2d', type=str, default='data/data_2d_h36m_gt.npz')
    parser.add_argument('--data_2d', type=str, default='data/data_2d_h36m_cpn_ft_h36m_dbb.npz')
    parser.add_argument('--subjects', type=str, default="['S9', 'S11']")
    parser.add_argument('--model_path', type=str, default='cpn.pth')
    parser.add_argument('--protocol', type=str, default='p2')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = args.data_3d
    dataset = Human36mDataset(dataset_path)
    keypoints_path = args.data_2d
    keypoints = np.load(keypoints_path, allow_pickle=True)
    # keypoints_metadata = keypoints['metadata'].item()
    # actions = dataset['S1'].keys()
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    # t = args.subjects
    all_actions = {}
    for a in dataset['S1'].keys():
        action_name = a.split(' ')[0]
        if action_name not in all_actions:
            all_actions[action_name] = []

    keypoints = keypoints['positions_2d'].item()
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions' not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions'].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
    subjects = keypoints.keys()
    for subject in subjects:
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                # kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
    subjects_eval = eval(args.subjects)
    x_train, y_train, cam_train = split_data(train_subjects, dataset, keypoints)
    print(subjects_eval)
    # model = Net(keypoints_num=17, n=3).cuda()
    num_view = 4
    model = IIVFormer(num_frame=num_view, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1).cuda()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    model.eval()
    loss_test = []
    for action in all_actions.keys():
        print('Evaluating action: {}'.format(action))
        # x, y = [], []
        x_eval, y_eval = [], []
        for subject in subjects_eval:
            for a in dataset[subject].keys():
                if action == a.split(' ')[0]:
                # if a.startswith(action): 
                    y_eval.append(dataset[subject][a]['positions'])
                    x_eval.append(np.concatenate(keypoints[subject][a], axis=2))
        x = np.concatenate(x_eval, axis=0)
        y = np.concatenate(y_eval, axis=0)
        y = y * 1000
        y = y - y[:, : 1, :]
        # x_train, x = normalation(x_train, x)
        x = (x - x_train.mean(axis=0)) / x_train.std(axis=0)


        x = x.reshape(-1, 17, 4, 2).transpose(0, 2, 1, 3)
        y = np.expand_dims(y, axis=1)

        eval_data = HumanDataset(x, y)
        eval_dataloader = torch.utils.data.DataLoader(eval_data, batch_size=1024, shuffle=False, num_workers=4)
        with torch.no_grad():
            N = 0
            loss_eval = 0
            for x, y in eval_dataloader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                pred = model(x)
                if args.protocol == 'p1':
                    loss = mpjpe(pred, y)
                    # print(loss.item())
                    loss_eval += loss.item() * y.shape[0]
                    # loss_eval += 
                elif args.protocol == 'p2':
                    pred = np.squeeze(pred.detach().cpu().numpy(), axis=1)
                    y = np.squeeze(y.detach().cpu().numpy(), axis=1)
                    loss = p_mpjpe(pred, y)
                    loss_eval += loss * y.shape[0]
                N += y.shape[0]
        loss_eval /= N
        loss_test.append(loss_eval)
        print('loss_eval on {}: is   {}'.format(action, loss_eval))
    
    for action in all_actions.keys():
        print('{:10}'.format(action), end='|')
    print()
    for loss in loss_test:
        loss = '{:.4f}'.format(loss)
        print('{:10}'.format(loss), end='|')
    print()
    print('loss_eval on all actions: is   {}'.format(np.mean(loss_test)))
if __name__ == '__main__':
    main()
