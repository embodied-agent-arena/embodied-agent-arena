#!/usr/bin/env python3
"""Build English display translations without changing evaluation prompts or results."""
import hashlib
import json
import re
from pathlib import Path

PUB = Path(__file__).resolve().parents[1] / 'public'
SOURCE = 'data/source-instructions.json'

DEPTH = """Using only this RGB image, estimate the depth of the visible surface at each sampled pixel.
The original image is W=1024 by H=768 pixels. The origin is the outer top-left edge, u increases to the right, and v increases downward; the top-left pixel center is (0.5,0.5). Locate points in these original coordinates even if the image is internally resized.
Depth is the camera-coordinate Z value in meters, with the origin at the camera's optical center and +Z along its forward optical axis. It is not Euclidean range, object size, or height above the table. Estimate the foremost surface visible at each pixel, not an occluded surface.
Camera intrinsics, pose, object sizes, and reference depths are not provided. Absolute scale is ambiguous in a single image; use perspective and common knowledge to give your most plausible estimate. Estimate each point independently rather than assuming a common plane.
Sample points in output order, each given as [u,v]:
{points}
Return only a JSON object with one field, depth_m: an array of 24 values in the same order as the sampled points. Use positive numbers, or null for a point whose depth cannot be estimated. Do not include explanations, extra fields, or Markdown."""

FOCAL = """Using only this image, estimate the focal length of the camera that captured it.
The original image is W=768 by H=1024 pixels. Report the answer in pixels at this original resolution, even if the visual input is internally resized.
The camera follows a pinhole perspective model with square pixels, zero skew, and no lens distortion. The principal point is known to be at the image center: cx=W/2, cy=H/2. Do not estimate or output the principal point.
Estimate only f, with fx=fy=f. This is focal length in pixels, not millimeters or field of view.
Sensor size, camera-to-scene distance, object sizes, camera pose, and calibration patterns are not provided. Use perspective cues to give your most plausible single estimate.
Return only a JSON object containing exactly one field, f, with a positive numeric value. Do not include explanations, extra fields, or Markdown."""

INTRINSICS = """Using only this image, estimate the intrinsics of the camera that captured it.
The original image is W=768 by H=1024 pixels. Report all values in pixels at this original resolution, even if the visual input is internally resized.
The camera follows a pinhole perspective model with square pixels, zero skew, and no lens distortion, so fx=fy. The principal point may be off-center; infer it from the image rather than assuming it is centered.
The origin is the outer top-left edge, u increases to the right, and v increases downward; the top-left pixel center is (0.5,0.5). The intrinsic matrix is K=[[fx,0,cx],[0,fy,cy],[0,0,1]].
fx and fy are focal lengths in pixels, not millimeters. cx and cy are principal-point coordinates, not the center of an object in the image.
Sensor size, camera-to-scene distance, object sizes, camera pose, and calibration patterns are not provided. Use perspective cues to give your most plausible single estimate.
Return only a JSON object containing exactly four numeric fields: fx, fy, cx, cy. Do not include explanations or Markdown."""

MOTION = {
    'fixed': (
        'Images A and B show a scene before and after, in that order. Camera position, orientation, focal length, and resolution are unchanged.\n'
        'Track the target object: {target}. It may translate on the table or remain stationary; it does not rotate, change size, or leave the table.\n'
        'Using only the two images, report:\n'
        '1. moved: whether the target changed position, as a Boolean.\n'
        '2. direction: the image-plane direction of the target from A to B, based on the displacement of its overall center. Right and down refer to image coordinates, not the object or world coordinate system. '
    ),
    'camera': (
        'Images A and B show a scene before and after, in that order. Camera position and orientation may change slightly; focal length and resolution are unchanged.\n'
        'Track the target object: {target}. It may translate relative to the table or remain stationary; it does not rotate, change size, or leave the table. Other objects remain fixed relative to the table.\n'
        'Use the table and other objects to separate camera motion from actual target motion. Using only the two images, report:\n'
        '1. moved: whether the target changed position relative to the table, as a Boolean. Apparent image motion caused only by camera motion does not count.\n'
        '2. direction: the direction of actual target motion in the viewpoint of image A. Imagine keeping the camera at its position and orientation in A, project both target positions into that view, and use the displacement of the overall center. Do not directly compare pixel coordinates across the two original images. Right and down refer to image A. '
    ),
    'real': (
        'Photographs A and B show a scene before and after, in that order. The camera may change; other objects on the table may also move, appear, or disappear.\n'
        'Track only the designated target: {target}. It may move relative to the table or remain stationary.\n'
        'Use the table and background elements that can be confirmed stationary to separate camera motion from actual target motion. Using only the two photographs, report:\n'
        '1. moved: whether the designated target changed position relative to the table, as a Boolean. Apparent image motion caused only by camera motion does not count.\n'
        '2. direction: the direction of actual target motion in the viewpoint of image A. Imagine keeping the camera at its position and orientation in A, project both target positions into that view, and use the displacement of the overall center. Right and down refer to image A, not the target itself. '
    ),
}
DIRECTIONS = ('Choose the closest of eight directions: right, down_right, down, down_left, left, up_left, up, up_right. Use none if there is no actual motion.\n'
              '3. distance_cm: the actual straight-line distance traveled by the target on the table, in centimeters, not pixel distance or camera displacement. Use everyday objects and scene cues to give your most plausible single estimate.\n'
              'If the target did not move, return moved=false, direction=none, and distance_cm=0. Otherwise, moved=true and distance_cm must be positive.\n'
              'Return only a JSON object with moved, direction, and distance_cm. Do not include explanations or Markdown.')

POSE = """Images A and B show the same static scene, in that order. All objects and lighting remain unchanged. Only the camera may translate or rotate, or it may remain stationary. Both images use the same camera with unchanged intrinsics and no zoom. Camera intrinsics, real object sizes, and metric calibration are not provided.
Using only these two images, estimate the 3D position and orientation of camera B relative to camera A.
Use a right-handed coordinate system centered at camera A's optical center: +X points right in the image, +Y points down, and +Z points forward from camera A into the scene.
translation_cm = [tx, ty, tz] gives camera B's optical center in A's coordinate frame, in centimeters. This is the camera's displacement, not apparent object motion or the extrinsic tvec that transforms points from A to B.
rotation_xyz_deg = [rx, ry, rz] gives B's orientation relative to A, using right-handed rotations about A's fixed X, Y, and Z axes in that order, in degrees. Specifically, R_A_from_B = Rz(rz) @ Ry(ry) @ Rx(rx), and p_A = R_A_from_B @ p_B + translation_cm, with p in centimeters. All six values are zero if the camera is unchanged.
Without real object dimensions, translation in centimeters depends on visual scale priors and cannot be uniquely recovered; still provide your most plausible numerical estimate. Do not assume all changes arise exclusively from translation or rotation.
Return only this JSON object, with three finite numbers in each array and no explanation or Markdown:
{"translation_cm": [tx, ty, tz], "rotation_xyz_deg": [rx, ry, rz]}"""

REFERENCE = [
    'The orange object is the target and the blue object is the reference. The reference is 10 cm tall. Both objects stand upright, and their nearest surfaces are at the same depth. Approximately how many times taller is the target than the reference? Return only JSON containing height_ratio.',
    'The orange object is the target and the blue object is the reference. The reference is 10 cm tall. Both objects stand upright, and their nearest surfaces are at the same depth. Estimate the target height in centimeters. Return only JSON containing height_cm.',
    'Compare the orange target objects in the two images. The blue reference in each image is 10 cm tall. Both objects stand upright, and their nearest surfaces are at the same depth within each image. What is the ratio of the target height in the second image to its height in the first? Return only JSON containing height_ratio_2_over_1.',
]

SIZE = """Using only this image, estimate the following real object dimensions in centimeters:
- Mug height, from the bottom to the rim: mug_height_cm.
- Total height of the capped water bottle, from the bottom to the top of the cap: bottle_height_cm.
- Length of the longer edge of the book cover: book_length_cm.
- Keyboard width from its leftmost to rightmost edge, including its casing: keyboard_width_cm.
Use the objects and scene context to give your most plausible single estimate for each dimension.
Return only a JSON object with these four fields and positive numeric values. Do not include explanations or Markdown."""

TARGETS = {
    'camera': ['computer mouse', 'book', 'computer mouse'],
    'camera_large': ['mug with a handle', 'computer mouse', 'book'],
    'fixed': ['capped water bottle', 'computer mouse', 'computer mouse'],
    'real': ['transparent glass jug with a green handle on the right side of the table',
             'blue letter C block on the right side of the table',
             'blue letter B block in the middle of the table'],
}


def translate(case, original):
    kind = case['id'].removeprefix('W1-w1_vlm_').rsplit('-', 1)[0]
    index = int(case['id'].rsplit('-', 1)[1]) - 1
    if kind == 'depth':
        points = next(line for line in original['question'].splitlines() if line.startswith('[['))
        assert len(json.loads(points)) == 24
        return DEPTH.format(points=points), 'Estimate surface depth at 24 sampled pixels.'
    if kind == 'focal':
        return FOCAL, 'Estimate focal length in pixels from a single image.'
    if kind == 'intrinsics':
        return INTRINSICS, 'Estimate focal length and principal point from a single image.'
    if kind.startswith('motion_'):
        mode = kind.removeprefix('motion_')
        template = MOTION['camera' if mode == 'camera_large' else mode]
        target = TARGETS[mode][index]
        short = f'Track the {target} and estimate its direction and travel distance.'
        return template.format(target=target) + DIRECTIONS, short
    if kind == 'pose':
        return POSE, "Estimate camera B's position and orientation in camera A's coordinate frame."
    if kind == 'reference':
        shorts = ['Compare target height against a 10 cm reference.',
                  'Estimate target height using a 10 cm reference.',
                  'Compare target heights across two images using a 10 cm reference.']
        return REFERENCE[index], shorts[index]
    if kind == 'size':
        return SIZE, 'Estimate mug, bottle, book, and keyboard dimensions in centimeters.'
    raise ValueError(f'No reviewed English translation for {case["id"]}')


def build():
    originals = json.loads((PUB / SOURCE).read_text())['cases']
    path = PUB / 'data/case-studies.json'
    cases = json.loads(path.read_text())
    count = 0
    for case in cases:
        if case['id'] not in originals:
            continue
        original = originals[case['id']]
        assert case['taskId'] == original['taskId']
        assert hashlib.sha256(original['question'].encode()).hexdigest() == original['questionSha256']
        question, short = translate(case, original)
        assert not re.search(r'[\u3400-\u9fff]', question + short)
        case.update(question=question, shortTask=short,
                    instructionKind='Task instructions (English translation)',
                    originalInstruction={'url': SOURCE, 'id': case['id'],
                                         'language': original['language'],
                                         'sha256': original['questionSha256']})
        count += 1
    assert count == len(originals) == 30
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    print(f'Built {count} English display translations; original prompts and evaluation results retained.')


if __name__ == '__main__':
    build()
