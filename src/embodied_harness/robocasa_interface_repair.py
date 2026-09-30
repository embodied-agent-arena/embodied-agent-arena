"""Shared RoboCasa/365 parameter contract and observation-backed action feedback."""
import inspect
import re
import math


def key(value):
    return re.sub(r'[^a-z0-9]+','',str(value).casefold())


def resolve_name(registry, value):
    if value is None:return None
    if isinstance(value,str) and value in registry:return value
    exact=[name for name,p in registry.items() if any(p.get(k) is not None and str(p[k])==str(value) for k in ('id','body_id'))]
    if len(exact)==1:return exact[0]
    canonical=[name for name in registry if key(name)==key(value)]
    if len(canonical)==1:return canonical[0]
    if len(canonical)>1:return None
    matches=[]
    for name,p in registry.items():
        aliases=[name,*p.get('agent_aliases',[])]
        if any(key(alias)==key(value) for alias in aliases):matches.append(name)
    # Never choose among multiple objects or guess a semantic target.
    return matches[0] if len(matches)==1 else None


def normalize_parameters(parameters, signature):
    result=dict(parameters);changed={}
    # The gateway may attach evidence to settle/read calls that do not consume it.
    if "evidence_handles" not in signature.parameters:
        for field in ("evidence_handle","evidence_handles"):
            if field in result:
                result.pop(field);changed[field]="not_consumed_by_this_primitive"
    groups=[('object','object_name'),('fixture','fixture_name'),('context','agent_context'),('evidence_handle','evidence_handles')]
    for old,new in groups:
        if old in result and new in signature.parameters:
            value=[result[old]] if old=='evidence_handle' else result[old]
            if new in result and result[new]!=value:raise ValueError('conflicting_parameters:'+old+','+new)
            result[new]=value;del result[old];changed[old]=new
    for old,choices in [('position',('world_position','target_position'))]:
        valid=[n for n in choices if n in signature.parameters]
        if old in result and len(valid)==1:
            new=valid[0]
            if new in result and result[new]!=result[old]:raise ValueError('conflicting_parameters:'+old+','+new)
            result[new]=result.pop(old);changed[old]=new
    if 'evidence_handles' in result and isinstance(result['evidence_handles'],str):
        result['evidence_handles']=[result['evidence_handles']]
        changed['evidence_handles']='single_handle_normalized_to_list'
    return result,changed


def feedback(output, before, after, *, require_lift=False):
    executed=output.get('execution_status') in ('stepped','executed')
    final=output.get('step_summary',{}).get('final',{})
    distance=final.get('distance_to_target')
    tolerance=output.get('tolerance',output.get('contact_tolerance'))
    arrived=(float(distance)<=float(tolerance)) if executed and isinstance(distance,(int,float)) and isinstance(tolerance,(int,float)) else None
    contact=after.get('gripper_object_contact');coupling=after.get('coupling',{})
    # Absolute tolerances alone can mark a stationary object as coupled during a tiny EE move.
    em=coupling.get('eef_motion') or 0.;om=coupling.get('object_motion') or 0.
    ce=coupling.get('coupling_error')
    confirmed=(coupling.get('coupled') is True and min(em,om)>=.005
               and isinstance(ce,(int,float)) and ce<=min(.02,.5*max(em,om)))
    eef_displacement=coupling.get('eef_displacement') or []
    object_displacement=coupling.get('object_displacement') or []
    lift_confirmed=(len(eef_displacement)>2 and len(object_displacement)>2
                    and eef_displacement[2]>=.03 and object_displacement[2]>=.03)
    grasp_confirmed=confirmed and (not require_lift or lift_confirmed)
    grasped=False if contact is False else True if contact is True and grasp_confirmed else None
    released=None
    if output.get('release_executed') is True:
        released=True if contact is False else False if grasped is True else None
    return dict(executed=executed,arrived=arrived,grasped=grasped,released=released,
                contact=contact,motion_coupling=coupling,motion_coupling_confirmed=confirmed,
                lift_confirmed=lift_confirmed if require_lift else None,
                release_command_executed=output.get('release_executed'),release_stable=output.get('release_stable'),
                unknown_is_not_success=True,official_task_success_not_included=True)


def invoke(handler, kwargs):
    from .schemas import PrimitiveResult
    from . import robocasa_agent_runtime as rc
    sig=inspect.signature(handler);backend=getattr(handler,'__self__',None)
    name=handler.__name__.removeprefix('_primitive_')
    try:
        args,changes=normalize_parameters(kwargs,sig)
        if not any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):sig.bind(**args)
    except (TypeError,ValueError) as exc:
        return PrimitiveResult(name=name,ok=False,error='invalid_parameters',output=dict(executed=False,message=str(exc),accepted_parameters=list(sig.parameters)))
    if backend is not None:
        registries={'object_name':backend._objects,'fixture_name':backend._fixtures,
                    'entity_name':{**backend._fixtures,**backend._objects},'target_name':{**backend._fixtures,**backend._objects}}
        for field,registry in registries.items():
            if args.get(field) is not None:
                canonical=resolve_name(registry,args[field])
                if canonical is not None:
                    if canonical!=args[field]:changes[field]={'supplied':args[field],'canonical':canonical}
                    args[field]=canonical
    if name=='place_robocasa_object_at' and backend is not None:
        stored=getattr(backend,'_held_object_handle',None)
        if args.get('held_object_handle') is None and args.get('object_name')==getattr(backend,'_held_object_name',None):
            args['held_object_handle']=stored
        if not stored or args.get('held_object_handle')!=stored or args.get('object_name')!=getattr(backend,'_held_object_name',None):
            return PrimitiveResult(name=name,ok=False,error='held_object_handle_required',
                                   output={'executed':False,'stage':'transport','retry_reason':'object_not_confirmed_held',
                                           'next_call':{'name':'observe_robocasa_kitchen_state','parameters':{}}})
    action=name.startswith(('grasp_','place_','move_','apply_','press_','open_','close_'))
    obj=args.get('object_name') or args.get('target_name')
    def observe(previous=None):
        if backend is None or obj not in backend._objects:return {}
        return rc._robocasa_live_transport_observation(backend._env,rc._read_current_observation(backend._env) or backend._last_obs,
                    object_name=obj,previous_state=previous,contact_tolerance=.08,gripper_closed_threshold=.035,coupling_tolerance=.02,minimum_coupling_motion=.005)
    before=observe() if action else {}
    result=handler(**args)
    result.metadata['parameter_normalization']=changes
    if action:
        state_after=observe(before)
        checked_after=(result.output.get('grasp_lift_evidence')
                       if name=='grasp_robocasa_object' and isinstance(result.output.get('grasp_lift_evidence'),dict)
                       else state_after)
        action_feedback=feedback(result.output,before,checked_after,require_lift=name=='grasp_robocasa_object')
        stage=('grasp' if name=='grasp_robocasa_object' else
               'release' if name=='place_robocasa_object_at' else 'approach')
        action_feedback.update(stage=stage,distance_m=result.output.get('distance_after'),
                               stable=result.output.get('release_stable') if stage=='release' else None)
        if name=='grasp_robocasa_object':
            has_lift_evidence=isinstance(result.output.get('grasp_lift_evidence'),dict)
            if not has_lift_evidence:
                action_feedback['grasped']=False if action_feedback['contact'] is False else None
                action_feedback['lift_confirmed']=False
            result.ok=bool(
                result.error is None and has_lift_evidence
                and action_feedback['executed'] and action_feedback['grasped'] is True
            )
            if result.ok:
                held=f"robocasa:held:{backend._observation_serial}:{args['object_name']}"
                backend._held_object_handle=held;backend._held_object_name=args['object_name']
                result.output['held_object_handle']=held
            else:
                result.error=result.error or (
                    'gripper_contact_absent' if action_feedback['contact'] is False
                    else 'object_did_not_lift' if action_feedback['lift_confirmed'] is False
                    else 'grasp_not_confirmed'
                )
                action_feedback.update(retry_reason=result.error,next_action='retry_grasp')
        elif name=='place_robocasa_object_at':
            action_feedback['holding_before']=bool(stored)
            action_feedback['xy_error_m']=result.output.get('xy_distance_after')
            result.ok=bool(action_feedback['executed'] and action_feedback['released'] is True and action_feedback['stable'] is True)
            if result.ok:
                backend._held_object_handle=None;backend._held_object_name=None
            else:
                result.error=result.error or 'release_not_confirmed'
                action_feedback.update(retry_reason=result.error,next_action='inspect_transport_state')
        result.output['action_feedback']=action_feedback
    if result.error=='visual_grounding_evidence_required':
        result.output['next_call_contract']=dict(required='evidence_handles',source='ground_robocasa_visual_target',
            input='Caller-selected segmentation_id or world_position plus prompt; no automatic semantic selection')
    return result


def compact_public_result(result):
    """Return only decision feedback to the agent; full result is already traced."""
    from .schemas import PrimitiveResult
    output=result.output
    if result.name.startswith(('grasp_robocasa','place_robocasa','move_robocasa','press_robocasa','open_robocasa','close_robocasa')):
        compact={key:output[key] for key in (
            'execution_status','action_feedback','held_object_handle','next_call','target_name',
            'release_executed','release_stable','xy_distance_after') if key in output}
        return PrimitiveResult(name=result.name,ok=result.ok,output=compact,
                               artifacts=list(result.artifacts),error=result.error,
                               metadata={'parameter_normalization':result.metadata.get('parameter_normalization',{})})
    return result
