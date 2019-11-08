import json
import pkg_resources
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps_per_json", default=1, help="number of steps in one json file, in PFNN, this number is 1",
                    type=int)
args = parser.parse_args()

if __name__ == '__main__':
    src_dbroot = pkg_resources.resource_filename(
        'phygamenn', 'data/PFNN/animations/')
    file_list = [f for f in sorted(list(os.listdir(src_dbroot))) if os.path.isfile(
        os.path.join(src_dbroot, f)) and f.endswith('.bvh') and 'rest' not in f]
    file_list = file_list[:1]

    deepm_dbroot = './data/dm/deepm/'
    json_step_dbroot = './data/dm/json_step/'
    deepm_json_files = [deepm_dbroot +
                        f.replace(".bvh", ".txt") for f in file_list]
    src_bvh_files = [src_dbroot + f for f in file_list]
    print(deepm_json_files)

    if not os.path.isdir(json_step_dbroot):
        os.mkdir(json_step_dbroot)

    for i, json_file in enumerate(deepm_json_files):
        bvh_file = src_bvh_files[i]
        file_name = file_list[i]
        with open(json_file, 'r') as f:
            json_dict = json.load(f)

        with open(bvh_file.replace('.bvh', '_footsteps.txt'), 'r') as f:
            footsteps = f.readlines()

        loop = json_dict['Loop']
        frames = json_dict['Frames']

        print(len(frames))
        print(footsteps)

        cnt = 0
        steps_per_json = args.steps_per_json
        window = 60
        for li in range(1, len(footsteps) - steps_per_json, steps_per_json):
            # curr, next = footsteps[li + 0].split(' '), footsteps[li + 1].split(' ')
            curr = []
            skip = False
            for index in range(steps_per_json + 1):
                curr.append(footsteps[li + index].split(' '))
                if len(curr[index]) == 3 and curr[index][2].strip().endswith('*'):
                    skip = True
            """ Ignore Cycles marked with '*' or not in range """

            if len(curr[steps_per_json]) < 2:
                skip = True
            if int(curr[0][0]) // 2 - window < 0:
                skip = True
            if int(curr[steps_per_json][0]) // 2 + window >= len(frames):
                skip = True

            if skip:
                continue

            slc = slice(int(curr[0][0]) // 2 - window,
                        int(curr[steps_per_json][0]) // 2 + window)
            output_name = file_name.replace('.bvh', '_')
            output_name = output_name + \
                str(int(curr[0][0])) + '_' + \
                str(int(curr[steps_per_json][0])) + '_json.txt'
            # output_name = os.path.join(json_step_dbroot, output_name)
            output = {}
            output["Loop"] = "none"
            output["Frames"] = frames[slc]
            with open(json_step_dbroot + output_name, 'w') as f:
                json.dump(output, f)
            cnt += 1
        print('%d segments in this bvh' % (cnt))
