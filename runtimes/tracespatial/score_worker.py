"""Private CPU scorer adapting pinned RoboTracer functions to one-case W1 results."""
import contextlib,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('robotracer_official',D/'upstream/Evaluation/summarize_acc.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def path_metrics(pred,gt,prefix):
 p=official.interpolate_trajectory_by_distance(pred,100);g=official.interpolate_trajectory_by_distance(gt,100)
 rmse,mae=official.calculate_rmse_mae(p,g)
 return {prefix+'frechet':official.discrete_frechet_distance(p,g),prefix+'hausdorff':official.hausdorff_distance(p,g),prefix+'rmse':rmse,prefix+'mae':mae}

def evaluate(answer,case,root):
 mode=case['benchmark'].rsplit('_',1)[1];dim=2 if mode=='2d' else 3
 if isinstance(answer,dict) and set(answer)=={'trajectory'}:answer=answer['trajectory']
 if isinstance(answer,str):
  try:answer=json.loads(answer)
  except ValueError:return {'submission_valid':False,'passed':False,'reason':'trajectory must be a JSON-compatible array'}
 valid=isinstance(answer,list) and 2<=len(answer)<=10 and all(isinstance(p,(list,tuple)) and len(p)==dim and all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in p) and (dim==2 or p[2]>0) for p in answer)
 if not valid:return {'submission_valid':False,'passed':False,'reason':'expected 2-10 finite points and positive depth in meters for 3D'}
 gt=case['gt'];root=Path(root);image=root/'raw_data'/gt['image_path'];depthpath=root/'raw_data'/gt['gt_depth_path'];maskpath=root/'raw_data'/gt['mask_path']
 receipt=json.loads((root/'download_receipt.json').read_text())
 for p in [image,depthpath,maskpath]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=receipt['files'][str(p.relative_to(root))]:raise ValueError('Pinned scoring asset changed: '+str(p.name))
 mask=np.asarray(Image.open(maskpath));depth=np.asarray(Image.open(depthpath));w,h=Image.open(image).size
 if mask.ndim==3:mask=mask[:,:,0]
 assert depth.shape[:2]==mask.shape[:2]==(h,w)
 K=official.extract_intrinsics_from_matrix(gt['gt_depth_intrinsics']);dims=np.array([w,h],np.float32);points=np.asarray(answer,np.float64);points[:,:2]=np.round(points[:,:2]/1000,6)
 truth=np.asarray(gt['trajectory'],np.float32);gt2=official.project_3d_to_2d(truth,K).astype(np.float32)
 raw2=points[:,:2]*dims
 end2=official.project_3d_bbox_to_2d(gt['bbox_center'],gt['bbox_extent'],gt['bbox_rotation'],K)
 endpts=raw2[-3:] if len(raw2)>=3 else raw2[-1:]
 metrics={'submission_valid':True,'task_family':gt['category'],'start_in_mask':bool(official.is_point_in_mask(raw2[0],mask)),'end_in_bbox_2d':bool(any(official.is_point_in_2d_bbox(p,end2) for p in endpts)),'xy_out_of_bounds':bool(np.any(points[:,:2]<0) or np.any(points[:,:2]>1)),'point_count':len(points),'mode':mode.upper(),'metric':'spatial_trace_'+mode}
 if mode=='2d':
  # Upstream interpolates in pixel space then normalizes the interpolants.
  p=official.interpolate_trajectory_by_distance(raw2,100)/dims;g=official.interpolate_trajectory_by_distance(gt2,100)/dims
  rmse,mae=official.calculate_rmse_mae(p,g)
  metrics.update(trace_2d_frechet=official.discrete_frechet_distance(p,g),trace_2d_hausdorff=official.hausdorff_distance(p,g),trace_2d_rmse=rmse,trace_2d_mae=mae,passed=None)
 else:
  pred3=official.backproject_to_3d(points,w,h,K);metrics.update(path_metrics(pred3,truth,'trace_3d_'))
  p3=official.interpolate_trajectory_by_distance(pred3,100);g3=official.interpolate_trajectory_by_distance(truth,100)
  p2=official.project_3d_to_2d(p3,K).astype(np.float32)/dims;g2=official.project_3d_to_2d(g3,K).astype(np.float32)/dims
  rmse,mae=official.calculate_rmse_mae(p2,g2);metrics.update(derived_2d_rmse=rmse,derived_2d_mae=mae,derived_2d_frechet=official.discrete_frechet_distance(p2,g2),derived_2d_hausdorff=official.hausdorff_distance(p2,g2))
  obj=official.create_object_pcd_from_mask(str(maskpath),str(depthpath),gt['gt_depth_intrinsics'])
  if obj is None or len(obj)==0:raise ValueError('Empty target point cloud; no success can be scored')
  start=float(min(np.min(np.linalg.norm(obj-pred3[0],axis=1)),np.linalg.norm(pred3[0]-truth[0])))
  ends=pred3[-3:] if len(pred3)>=3 else pred3[-1:];end=float(min(official.point_to_box_distance(p,np.array(gt['bbox_center'],np.float32),np.array(gt['bbox_extent'],np.float32),np.array(gt['bbox_rotation'],np.float32)) for p in ends))
  grid=official.create_occupancy_grid_from_tsdf(depth.astype(np.float32),mask.astype(np.uint8),gt['gt_depth_intrinsics'])
  ratios=official.calculate_trajectory_collisions(grid,obj,pred3)
  if ratios is None:raise ValueError('Collision check unavailable; do not assume no collision')
  collision=any(r>.20 for r in ratios)
  metrics.update(start_distance_m=start,end_distance_m=end,start_success=start<.20,end_success=end<.20,collision=bool(collision),max_collision_ratio=float(max(ratios)),no_collision=not collision,passed=bool(start<.20 and end<.20 and not collision))
 # Unfiltered per-case values are always retained. Upstream aggregate silently filters DFD>100.
 metrics['upstream_path_aggregate_eligible']=metrics['trace_'+mode+'_frechet']<=100
 return metrics

if __name__=='__main__':
 request=json.load(sys.stdin)
 with contextlib.redirect_stdout(sys.stderr):result=evaluate(request['answer'],request['case'],request['data_root'])
 print(json.dumps(result,allow_nan=False))
