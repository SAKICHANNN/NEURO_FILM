"""Frozen P4AN audit of a low-frequency-controlled Gaussian copula."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import spearmanr
from src.eval.physical_callier_source import hash_file
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    render_balanced_derivative_structure,
    render_balanced_derivative_structure_region,
    render_derivative_conditioned_structure,
)
from src.film_physics.structure_compiler import balanced_correlated_normal_region, correlated_normal_region

SCHEMA="neuro_film.u6_p4an_balanced_gaussian_copula_structure_contract.v1"
REPORT_SCHEMA="neuro_film.u6_p4an_balanced_gaussian_copula_structure_report.v1"
class BalancedGaussianCopulaError(RuntimeError): pass
def _canonical(v:Any)->bytes:return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
def load_contract(path:Path)->dict[str,Any]:
 v=json.loads(path.read_text(encoding="utf-8"));m=v.get("model",{});e=v.get("evaluation",{});g=v.get("gates",{})
 if (v.get("schema")!=SCHEMA or m.get("block_shape")!=[2,2] or m.get("peak_density_variance")!=.0001 or m.get("correlation_sigma_pixels")!=.65 or m.get("layer_seeds")!=[260831,260837,260851] or m.get("fitted_parameters")!=0 or m.get("sample_or_image_centering_allowed") or e.get("shape")!=[512,768] or e.get("low_frequency_block_sizes")!=[4,8] or e.get("row_partitions")!=[31,127] or e.get("runs")!=2 or e.get("post_result_retuning_allowed") or g.get("minimum_flat_transmittance_error_improvement_over_p4af")!=.5 or g.get("maximum_low_frequency_block_mean_std_ratio_to_p4af")!=.5 or g.get("maximum_checker_frequency_power_ratio_to_neighborhood")!=3.0 or g.get("maximum_isolated_standardized_excursions")!=0 or not g.get("no_parameter_fit")): raise BalancedGaussianCopulaError("P4AN frozen contract drift")
 return v
def _parents(c:dict[str,Any],root:Path)->tuple[dict[str,Any],dict[str,str]]:
 ids={}
 for stem in ("p4am_decision","p4af_decision","sensitometry_contract"):
  rel=Path(c["parents"][f"{stem}_path"])
  if rel.is_absolute() or ".." in rel.parts: raise BalancedGaussianCopulaError("P4AN parent path escaped root")
  path=root/rel; actual=hash_file(path)
  if actual!=c["parents"][f"{stem}_sha256"]: raise BalancedGaussianCopulaError(f"P4AN parent mismatch: {stem}")
  ids[stem]=actual
 return json.loads((root/c["parents"]["sensitometry_contract_path"]).read_text(encoding="utf-8")),ids
def _profile(c:dict[str,Any])->DerivativeConditionedStructureProfile:
 m=c["model"];return DerivativeConditionedStructureProfile(float(m["peak_density_variance"]),float(m["correlation_sigma_pixels"]),tuple(m["layer_seeds"]),float(m["variance_normalization_domain"][0]),float(m["variance_normalization_domain"][1]),int(m["variance_normalization_samples"]))
def _lag_xy(v:np.ndarray)->tuple[float,float]: return float(np.corrcoef(v[:,:-1].ravel(),v[:,1:].ravel())[0,1]),float(np.corrcoef(v[:-1].ravel(),v[1:].ravel())[0,1])
def _block_std(v:np.ndarray,f:int)->float:
 h=(v.shape[0]//f)*f;w=(v.shape[1]//f)*f;return float(np.std(v[:h,:w].reshape(h//f,f,w//f,f).mean((1,3))))
def _checker_ratio(v:np.ndarray)->float:
 power=np.abs(np.fft.fft2(v-np.mean(v)))**2;h,w=power.shape;ratios=[]
 for y,x in ((0,w//2),(h//2,0),(h//2,w//2)):
  neighborhood=power[np.ix_([(y+d)%h for d in range(-2,3)],[(x+d)%w for d in range(-2,3)])].ravel();center=power[y,x];others=np.delete(neighborhood,12);ratios.append(float(center/max(float(np.median(others)),1e-30)))
 return max(ratios)
def _isolated(z:np.ndarray)->int:
 kernel=np.ones((3,3),dtype=np.int16);kernel[1,1]=0;total=0
 for ch in range(3):
  mag=np.abs(z[...,ch]);neighbors=convolve((mag>=3).astype(np.int16),kernel,mode="constant");total+=int(np.count_nonzero((mag>=6)&(neighbors<2)))
 return total
def evaluate_structure(c:dict[str,Any],root:Path)->dict[str,Any]:
 sens,ids=_parents(c,root);op=build_operator(sens);p=_profile(c);e=c["evaluation"];g=c["gates"];h,w=e["shape"];b=int(e["interior_border_pixels"]);sl=np.s_[b:-b,b:-b,:]
 flat_err=[];base_err=[];var_err=[];lag=[];isolated=0;unchanged=True
 for level in e["flat_exposure_levels"]:
  src=np.full((h,w,3),level,dtype=np.float64);copy=src.copy();mean,shape=derivative_variance_shape(src,op,p);var=p.peak_density_variance*shape;candidate=render_balanced_derivative_structure(src,op,p);base=render_derivative_conditioned_structure(src,op,p);target=10**(-mean[0,0]);flat_err.append(float(np.max(np.abs(np.mean(candidate.transmittance[sl],axis=(0,1))-target))));base_err.append(float(np.max(np.abs(np.mean(base.transmittance[sl],axis=(0,1))-target))));res=candidate.density[sl].astype(np.float64)-mean[sl];active=var[0,0]>0;observed=np.mean(res*res,axis=(0,1));var_err.extend((np.abs(observed[active]-var[0,0,active])/var[0,0,active]).tolist());standard=res/np.sqrt(var[0,0]);isolated+=_isolated(standard);lag.extend(_lag_xy(standard[...,ch]) for ch in range(3));unchanged &= np.array_equal(src,copy)
 flat=max(flat_err);base_flat=max(base_err);flat_gain=1-flat/base_flat
 rampv=np.geomspace(*e["ramp_exposure_interval"],w);ramp=np.broadcast_to(rampv[None,:,None],(h,w,3)).copy();mean,shape=derivative_variance_shape(ramp,op,p);candidate=render_balanced_derivative_structure(ramp,op,p);second=render_balanced_derivative_structure(ramp,op,p);repeat=np.array_equal(candidate.density,second.density) and np.array_equal(candidate.transmittance,second.transmittance);assembled=np.empty_like(candidate.density);assembled_t=np.empty_like(candidate.transmittance);partition=True
 for rows in e["row_partitions"]:
  for y in range(0,h,rows):
   r=render_balanced_derivative_structure_region(ramp,op,p,origin_yx=(y,0),shape=(min(rows,h-y),w));assembled[y:y+r.density.shape[0]]=r.density;assembled_t[y:y+r.density.shape[0]]=r.transmittance
  partition &= np.array_equal(candidate.density,assembled) and np.array_equal(candidate.transmittance,assembled_t)
 observed=[];expected=[];edges=np.linspace(0,w,e["ramp_bins"]+1,dtype=int);res=candidate.density.astype(np.float64)-mean
 for ch in range(3):
  for x0,x1 in zip(edges[:-1],edges[1:]): observed.append(float(np.mean(res[b:-b,x0:x1,ch]**2)));expected.append(float(p.peak_density_variance*np.mean(shape[b:-b,x0:x1,ch])))
 rank=float(spearmanr(expected,observed).statistic)
 balanced=balanced_correlated_normal_region((h,w),origin_yx=(0,0),shape=(h,w),sigma=p.correlation_sigma_pixels,seed=p.layer_seeds[0])[b:-b,b:-b];baseline=correlated_normal_region((h,w),origin_yx=(0,0),shape=(h,w),sigma=p.correlation_sigma_pixels,seed=p.layer_seeds[0])[b:-b,b:-b];ratios=[_block_std(balanced,f)/_block_std(baseline,f) for f in e["low_frequency_block_sizes"]];lx,ly=_lag_xy(balanced)
 metrics={"maximum_empirical_flat_transmittance_mean_absolute_error":flat,"maximum_p4af_flat_transmittance_mean_absolute_error":base_flat,"flat_transmittance_error_improvement_over_p4af":flat_gain,"maximum_density_variance_relative_error":max(var_err),"ramp_variance_shape_spearman":rank,"lag1_x":lx,"lag1_y":ly,"minimum_lag1_autocorrelation":min(lx,ly),"maximum_lag1_autocorrelation":max(lx,ly),"lag1_anisotropy":abs(lx-ly),"low_frequency_block_mean_std_ratios_to_p4af":ratios,"maximum_low_frequency_block_mean_std_ratio_to_p4af":max(ratios),"checker_frequency_power_ratio_to_neighborhood":_checker_ratio(balanced),"isolated_standardized_excursions":isolated,"density_minimum":float(np.min(candidate.density)),"transmittance_minimum":float(np.min(candidate.transmittance)),"transmittance_maximum":float(np.max(candidate.transmittance)),"repeat_exact":bool(repeat),"row_partition_exact":bool(partition),"input_unchanged":bool(unchanged)}
 checks={"flat_transmittance_mean":flat<=g["maximum_empirical_flat_transmittance_mean_absolute_error"],"flat_improvement":flat_gain>=g["minimum_flat_transmittance_error_improvement_over_p4af"],"density_variance":metrics["maximum_density_variance_relative_error"]<=g["maximum_density_variance_relative_error"],"variance_shape":rank>=g["minimum_ramp_variance_shape_spearman"],"lag1_range":min(lx,ly)>=g["minimum_lag1_autocorrelation"] and max(lx,ly)<=g["maximum_lag1_autocorrelation"],"lag1_anisotropy":abs(lx-ly)<=g["maximum_lag1_anisotropy"],"low_frequency":max(ratios)<=g["maximum_low_frequency_block_mean_std_ratio_to_p4af"],"checker_frequency":metrics["checker_frequency_power_ratio_to_neighborhood"]<=g["maximum_checker_frequency_power_ratio_to_neighborhood"],"isolated":isolated<=g["maximum_isolated_standardized_excursions"],"physical_domain":metrics["density_minimum"]>=0 and metrics["transmittance_minimum"]>0 and metrics["transmittance_maximum"]<=1,"repeat_exact":bool(repeat),"row_partition_exact":bool(partition),"input_unchanged":bool(unchanged),"no_parameter_fit":True}
 stable={"experiment_id":c["experiment_id"],"parent_identities":ids,"metrics":metrics,"checks":checks,"passed":all(checks.values())};branch="pass" if stable["passed"] else "fail";return {"schema":REPORT_SCHEMA,**stable,"stable_evidence_id":hashlib.sha256(_canonical(stable)).hexdigest(),"decision":c["branch_rule"][branch],"claim_ceiling":c["claim_ceiling"]}
__all__=["BalancedGaussianCopulaError","REPORT_SCHEMA","SCHEMA","evaluate_structure","load_contract"]
