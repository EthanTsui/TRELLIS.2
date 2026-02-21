#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer for TRELLIS.2 3D Generation Quality.

Evolves parameter configurations using:
- Tournament selection (elite preservation)
- Uniform crossover (blend parent parameters)
- Gaussian mutation (bounded perturbation)
- Fitness-based ranking

Integrates with the research knowledge base:
- Reads population from research/population.json
- Writes results to research/experiment_log.json
- Updates population with new generation
- Logs to research/changelog.md

Usage:
    python optimization/scripts/genetic_optimizer.py --generations 5 --population-size 8
    python optimization/scripts/genetic_optimizer.py --evaluate-only  # Just evaluate queued individuals
    python optimization/scripts/genetic_optimizer.py --evolve-only    # Just create next generation
"""

import os
import sys
import json
import time
import math
import random
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

# Add TRELLIS.2 to path
TRELLIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TRELLIS_ROOT)

import numpy as np
import torch
from PIL import Image

# Research knowledge base paths
RESEARCH_DIR = os.path.join(TRELLIS_ROOT, 'optimization', 'research')
POPULATION_PATH = os.path.join(RESEARCH_DIR, 'population.json')
EXPERIMENT_LOG_PATH = os.path.join(RESEARCH_DIR, 'experiment_log.json')
CHANGELOG_PATH = os.path.join(RESEARCH_DIR, 'changelog.md')


# ============================================================
# KNOWLEDGE BASE I/O
# ============================================================

def load_population() -> Dict:
    """Load population from knowledge base."""
    with open(POPULATION_PATH, 'r') as f:
        return json.load(f)


def save_population(pop: Dict):
    """Save population to knowledge base."""
    with open(POPULATION_PATH, 'w') as f:
        json.dump(pop, f, indent=2, default=str)


def load_experiment_log() -> Dict:
    """Load experiment log from knowledge base."""
    with open(EXPERIMENT_LOG_PATH, 'r') as f:
        return json.load(f)


def save_experiment_log(log: Dict):
    """Save experiment log to knowledge base."""
    with open(EXPERIMENT_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def append_changelog(text: str):
    """Append entry to changelog."""
    with open(CHANGELOG_PATH, 'a') as f:
        f.write(f"\n{text}\n")


# ============================================================
# GENETIC OPERATORS
# ============================================================

def tournament_select(individuals: List[Dict], tournament_size: int = 3) -> Dict:
    """Select an individual via tournament selection."""
    evaluated = [ind for ind in individuals if ind.get('fitness') is not None]
    if not evaluated:
        return random.choice(individuals)

    tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
    return max(tournament, key=lambda x: x['fitness']['overall'])


def uniform_crossover(parent1: Dict, parent2: Dict, bounds: Dict) -> Dict:
    """Create offspring via uniform crossover of two parents."""
    child_params = {}
    for param_name in bounds:
        if param_name in parent1['params'] and param_name in parent2['params']:
            # 50/50 chance of inheriting from either parent
            if random.random() < 0.5:
                child_params[param_name] = parent1['params'][param_name]
            else:
                child_params[param_name] = parent2['params'][param_name]
        elif param_name in parent1['params']:
            child_params[param_name] = parent1['params'][param_name]
        elif param_name in parent2['params']:
            child_params[param_name] = parent2['params'][param_name]

    return child_params


def blend_crossover(parent1: Dict, parent2: Dict, bounds: Dict, alpha: float = 0.3) -> Dict:
    """Create offspring via BLX-alpha crossover (blend between parents)."""
    child_params = {}
    for param_name, bound in bounds.items():
        v1 = parent1['params'].get(param_name)
        v2 = parent2['params'].get(param_name)
        if v1 is None and v2 is None:
            # Neither parent has this param — use midpoint of bounds as default
            if bound['type'] == 'float':
                child_params[param_name] = round((bound['min'] + bound['max']) / 2, 4)
            else:
                child_params[param_name] = int((bound['min'] + bound['max']) / 2)
            continue
        if v1 is None or v2 is None:
            child_params[param_name] = v1 if v1 is not None else v2
            continue

        if bound['type'] == 'float':
            lo = min(v1, v2) - alpha * abs(v2 - v1)
            hi = max(v1, v2) + alpha * abs(v2 - v1)
            lo = max(lo, bound['min'])
            hi = min(hi, bound['max'])
            value = random.uniform(lo, hi)
            # Snap to step
            step = bound.get('step', 0.01)
            value = round(value / step) * step
            child_params[param_name] = round(value, 4)
        elif bound['type'] == 'int':
            lo = min(v1, v2)
            hi = max(v1, v2)
            value = random.randint(lo, hi)
            step = bound.get('step', 1)
            value = round(value / step) * step
            child_params[param_name] = int(value)

    return child_params


def _mutate_param(value, bound, sigma: float = 0.2):
    """Apply Gaussian noise to a single parameter value."""
    param_range = bound['max'] - bound['min']
    if bound['type'] == 'float':
        noise = random.gauss(0, sigma * param_range)
        new_value = value + noise
        new_value = max(bound['min'], min(bound['max'], new_value))
        step = bound.get('step', 0.01)
        new_value = round(new_value / step) * step
        return round(new_value, 4)
    elif bound['type'] == 'int':
        noise = random.gauss(0, sigma * param_range)
        new_value = int(round(value + noise))
        new_value = max(bound['min'], min(bound['max'], new_value))
        step = bound.get('step', 1)
        new_value = round(new_value / step) * step
        return int(new_value)
    return value


def gaussian_mutate(params: Dict, bounds: Dict, mutation_rate: float = 0.15, sigma: float = 0.2) -> Dict:
    """Apply Gaussian mutation to parameters. Guarantees at least 1 param is mutated."""
    mutated = deepcopy(params)
    mutated_any = False

    for param_name, bound in bounds.items():
        if param_name not in mutated or mutated[param_name] is None:
            if bound['type'] == 'float':
                mutated[param_name] = round((bound['min'] + bound['max']) / 2, 4)
            else:
                mutated[param_name] = int((bound['min'] + bound['max']) / 2)
        if random.random() > mutation_rate:
            continue

        mutated[param_name] = _mutate_param(mutated[param_name], bound, sigma)
        mutated_any = True

    # Guarantee at least one parameter is mutated to avoid duplicates
    if not mutated_any:
        param_name = random.choice(list(bounds.keys()))
        bound = bounds[param_name]
        if param_name not in mutated or mutated[param_name] is None:
            if bound['type'] == 'float':
                mutated[param_name] = round((bound['min'] + bound['max']) / 2, 4)
            else:
                mutated[param_name] = int((bound['min'] + bound['max']) / 2)
        mutated[param_name] = _mutate_param(mutated[param_name], bound, sigma)

    return mutated


def create_next_generation(pop: Dict) -> List[Dict]:
    """Create a new generation from the current population."""
    bounds = pop['parameter_bounds']
    individuals = pop['individuals']
    pop_size = pop['population_size']
    elite_ratio = pop.get('elite_ratio', 0.25)
    mutation_rate = pop.get('mutation_rate', 0.15)
    crossover_rate = pop.get('crossover_rate', 0.6)
    generation = pop['generation'] + 1

    # Sort by fitness (evaluated individuals first, then by score)
    evaluated = [ind for ind in individuals if ind.get('fitness') is not None]
    evaluated.sort(key=lambda x: x['fitness']['overall'], reverse=True)

    if not evaluated:
        print("No evaluated individuals. Cannot evolve.")
        return individuals

    # Elite selection: keep top N
    n_elite = max(1, int(pop_size * elite_ratio))
    elites = evaluated[:n_elite]
    new_gen = []

    # Preserve elites
    for elite in elites:
        preserved = deepcopy(elite)
        preserved['generation'] = generation
        preserved['origin'] = f'elite_from_gen{generation-1}'
        preserved['status'] = 'evaluated'  # Already tested
        new_gen.append(preserved)

    # Generate offspring to fill remaining slots
    n_offspring = pop_size - len(new_gen)
    for i in range(n_offspring):
        if random.random() < crossover_rate and len(evaluated) >= 2:
            # Crossover
            p1 = tournament_select(evaluated)
            p2 = tournament_select(evaluated)
            # Avoid self-crossover
            attempts = 0
            while p1['id'] == p2['id'] and attempts < 5:
                p2 = tournament_select(evaluated)
                attempts += 1

            child_params = blend_crossover(p1, p2, bounds)
            origin = f'crossover({p1["id"]},{p2["id"]})'
        else:
            # Mutation from random evaluated individual
            parent = tournament_select(evaluated)
            child_params = deepcopy(parent['params'])
            origin = f'mutation({parent["id"]})'

        # Apply mutation
        child_params = gaussian_mutate(child_params, bounds, mutation_rate)

        child = {
            'id': f'gen{generation}-{i+1:03d}',
            'generation': generation,
            'origin': origin,
            'params': child_params,
            'fitness': None,
            'seed': random.randint(1, 10000),
            'status': 'queued',
        }
        new_gen.append(child)

    return new_gen


# ============================================================
# EVALUATION (reuses auto_optimize infrastructure)
# ============================================================

def evaluate_individual(
    individual: Dict,
    pipeline,
    evaluator,
    examples: Dict,
    fixed_params: Dict,
    render_dir: Path,
    glb_dir: Path,
) -> Dict:
    """Evaluate a single individual's fitness."""
    from evaluate import render_mesh_at_views

    # Merge fixed params + individual params into full config
    config = deepcopy(fixed_params)
    config.update(individual['params'])
    config['seed'] = individual.get('seed', 42)

    # Import needed constants
    from auto_optimize import TEST_EXAMPLES, VIEW_ANGLES, UPLOAD_DIR

    total_score = 0
    n_examples = 0
    per_example = {}
    per_dimension = {'silhouette': 0, 'contour': 0, 'color': 0, 'detail': 0,
                     'artifacts': 0, 'smoothness': 0, 'coherence': 0}

    for eid, example in examples.items():
        try:
            print(f"    Example {eid}: {example['info']['name']}...")

            # Generate
            t0 = time.time()
            outputs = pipeline.run(
                example['pipeline_input'],
                seed=config['seed'],
                preprocess_image=True,
                # NOTE: APG/beta schedule only safe for texture stage (Stage 3).
                # Shape stages (1 & 2) must use standard CFG -- APG's orthogonal
                # injection corrupts binary occupancy and SDF geometry features.
                sparse_structure_sampler_params={
                    'steps': config.get('ss_sampling_steps', 12),
                    'guidance_strength': config.get('ss_guidance_strength', 7.5),
                    'guidance_rescale': config.get('ss_guidance_rescale', 0.7),
                    'rescale_t': config.get('ss_rescale_t', 5.0),
                    'zero_init_steps': config.get('zero_init_steps', 0),
                },
                shape_slat_sampler_params={
                    'steps': config.get('shape_slat_sampling_steps', 12),
                    'guidance_strength': config.get('shape_slat_guidance_strength', 7.5),
                    'guidance_rescale': config.get('shape_slat_guidance_rescale', 0.5),
                    'rescale_t': config.get('shape_slat_rescale_t', 3.0),
                    'zero_init_steps': config.get('zero_init_steps', 0),
                },
                tex_slat_sampler_params={
                    'steps': config.get('tex_slat_sampling_steps', 12),
                    'guidance_strength': config.get('tex_slat_guidance_strength', 5.0),
                    'guidance_rescale': config.get('tex_slat_guidance_rescale', 0.85),
                    'guidance_interval': [0.0, 1.0],
                    'rescale_t': config.get('tex_slat_rescale_t', 3.0),
                    'cfg_mode': config.get('cfg_mode', 'standard'),
                    'apg_alpha': config.get('apg_alpha', 0.3),
                    'zero_init_steps': config.get('zero_init_steps', 0),
                    'guidance_schedule': config.get('guidance_schedule', 'binary'),
                    'guidance_beta_a': config.get('guidance_beta_a', 2.0),
                    'guidance_beta_b': config.get('guidance_beta_b', 5.0),
                },
                pipeline_type={
                    '512': '512',
                    '1024': '1024_cascade',
                    '1536': '1536_cascade',
                }.get(config.get('resolution', '1024'), '1024_cascade'),
                return_latent=True,
                multiview_mode=config.get('multiview_mode', 'concat'),
                texture_multiview_mode=config.get('texture_multiview_mode', 'tapa'),
            )
            meshes, latents = outputs
            mesh = meshes[0]
            gen_time = time.time() - t0

            # Render + evaluate
            view_names = example['info']['view_names']
            view_angles = {vn: VIEW_ANGLES[vn] for vn in view_names if vn in VIEW_ANGLES}

            rendered = render_mesh_at_views(mesh, view_angles, resolution=512, r=2.0, fov=36.0)
            # Pass mesh geometry for mesh-level smoothness evaluation
            mesh_verts = mesh.vertices if hasattr(mesh, 'vertices') else None
            mesh_faces_t = mesh.faces if hasattr(mesh, 'faces') else None
            scores = evaluator.evaluate(
                rendered, example['reference_views'],
                mesh_vertices=mesh_verts, mesh_faces=mesh_faces_t,
            )

            overall = scores.get('overall', 0)
            per_example[str(eid)] = {
                'overall': round(overall, 2),
                'silhouette': round(scores.get('silhouette', 0), 2),
                'contour': round(scores.get('contour', 0), 2),
                'color': round(scores.get('color', 0), 2),
                'detail': round(scores.get('detail', 0), 2),
                'artifacts': round(scores.get('artifacts', 0), 2),
                'smoothness': round(scores.get('smoothness', 0), 2),
                'coherence': round(scores.get('coherence', 0), 2),
            }
            total_score += overall
            n_examples += 1

            for dim in per_dimension:
                per_dimension[dim] += scores.get(dim, 0)

            # Save renders
            for vname, rimg in rendered.items():
                rpath = render_dir / f'{individual["id"]}_ex{eid}_{vname}.png'
                Image.fromarray(rimg).save(str(rpath))

            print(f"    → Score: {overall:.1f}/100 (gen: {gen_time:.0f}s)")

        except Exception as e:
            print(f"    ERROR on example {eid}: {e}")
            traceback.print_exc()
            per_example[str(eid)] = {'overall': 0, 'error': str(e)}

            # Check if CUDA context is corrupted (illegal memory access)
            if 'illegal memory access' in str(e).lower() or 'cuda error' in str(e).lower():
                print(f"    !! CUDA context likely corrupted, attempting recovery...")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                # Try a small allocation to test CUDA health
                try:
                    test = torch.zeros(1, device='cuda')
                    del test
                    print(f"    !! CUDA recovery successful, continuing...")
                except Exception as cuda_err:
                    print(f"    !! CUDA recovery FAILED: {cuda_err}")
                    print(f"    !! Skipping remaining examples for this individual")
                    break

        torch.cuda.empty_cache()

    # Compute fitness
    avg_score = total_score / max(n_examples, 1)
    for dim in per_dimension:
        per_dimension[dim] /= max(n_examples, 1)
        per_dimension[dim] = round(per_dimension[dim], 2)

    fitness = {
        'overall': round(avg_score, 2),
        'per_example': per_example,
        'per_dimension': per_dimension,
    }

    return fitness


# ============================================================
# MAIN LOOP
# ============================================================

class GeneticOptimizer:
    """Main GA optimization loop."""

    def __init__(self, output_dir: str, example_ids: List[int]):
        self.output_dir = Path(output_dir)
        self.example_ids = example_ids
        self.pipeline = None
        self.evaluator = None

        # Create output directories
        self.render_dir = self.output_dir / 'renders'
        self.glb_dir = self.output_dir / 'glb_outputs'
        for d in [self.render_dir, self.glb_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_pipeline(self):
        """Load TRELLIS.2 pipeline and evaluator."""
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from evaluate import QualityEvaluator

        print("Loading TRELLIS.2 pipeline...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        self.pipeline.cuda()
        self.evaluator = QualityEvaluator(device='cuda')
        print("Pipeline loaded.")

    def load_examples(self) -> Dict:
        """Load test examples."""
        from auto_optimize import TEST_EXAMPLES, UPLOAD_DIR
        from evaluate import split_input_image

        examples = {}
        for eid in self.example_ids:
            if eid not in TEST_EXAMPLES:
                continue
            info = TEST_EXAMPLES[eid]
            img_path = os.path.join(UPLOAD_DIR, info['image'])
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert('RGBA')
            views = split_input_image(image, info['layout'], info['view_names'])
            pipeline_input = {vname: Image.fromarray(varr) for vname, varr in views.items()}

            examples[eid] = {
                'info': info,
                'reference_views': views,
                'pipeline_input': pipeline_input,
            }
            print(f"  Loaded example {eid}: {info['name']}")

        return examples

    def evaluate_queued(self, examples: Dict):
        """Evaluate all queued individuals in the population."""
        pop = load_population()
        log = load_experiment_log()

        queued = [ind for ind in pop['individuals'] if ind['status'] == 'queued']
        if not queued:
            print("No queued individuals to evaluate.")
            return

        print(f"\nEvaluating {len(queued)} queued individuals...")

        for i, individual in enumerate(queued):
            print(f"\n  [{i+1}/{len(queued)}] Evaluating {individual['id']} "
                  f"(origin: {individual['origin']})...")

            try:
                fitness = evaluate_individual(
                    individual, self.pipeline, self.evaluator, examples,
                    pop['fixed_params'], self.render_dir, self.glb_dir,
                )
            except Exception as eval_err:
                print(f"  !! FATAL error evaluating {individual['id']}: {eval_err}")
                # Assign zero fitness so it's marked evaluated (won't block)
                fitness = {
                    'overall': 0,
                    'per_example': {},
                    'per_dimension': {},
                    'error': str(eval_err),
                }
                # Try to recover CUDA (all wrapped in try/except to avoid secondary crash)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Update individual
            individual['fitness'] = fitness
            individual['status'] = 'evaluated'

            # Log experiment
            exp_entry = {
                'id': f"exp-ga-{individual['id']}",
                'run_id': f'ga_gen{individual["generation"]}',
                'timestamp': datetime.now().isoformat(),
                'variation': individual['id'],
                'hypothesis_id': None,
                'config': {**pop['fixed_params'], **individual['params']},
                'scores': {
                    'avg': fitness['overall'],
                    'per_example': fitness['per_example'],
                },
                'notes': f"GA individual {individual['id']}, origin: {individual['origin']}",
            }
            log['experiments'].append(exp_entry)

            print(f"  → Fitness: {fitness['overall']:.1f}")

            # Save after each evaluation (fault tolerance)
            save_population(pop)
            save_experiment_log(log)

        # Summary
        evaluated = [ind for ind in pop['individuals'] if ind.get('fitness')]
        evaluated.sort(key=lambda x: x['fitness']['overall'], reverse=True)
        print(f"\nEvaluation complete. Top 3:")
        for ind in evaluated[:3]:
            print(f"  {ind['id']}: {ind['fitness']['overall']:.1f} "
                  f"(origin: {ind['origin']})")

    def evolve(self):
        """Create next generation."""
        pop = load_population()

        evaluated = [ind for ind in pop['individuals'] if ind.get('fitness')]
        if len(evaluated) < 2:
            print("Need at least 2 evaluated individuals to evolve.")
            return

        old_gen = pop['generation']
        print(f"\nEvolving from generation {old_gen}...")

        # Create next generation
        new_individuals = create_next_generation(pop)
        pop['individuals'] = new_individuals
        pop['generation'] = old_gen + 1

        save_population(pop)

        # Log to changelog
        queued = [ind for ind in new_individuals if ind['status'] == 'queued']
        elites = [ind for ind in new_individuals if 'elite' in ind.get('origin', '')]
        best = max(evaluated, key=lambda x: x['fitness']['overall'])

        changelog_entry = f"""
## Generation {pop['generation']} — Evolved ({datetime.now().strftime('%Y-%m-%d %H:%M')})

### Selection Summary
- Previous generation: {old_gen} ({len(evaluated)} evaluated)
- Best fitness: {best['fitness']['overall']:.1f} ({best['id']})
- Elites preserved: {len(elites)}
- New offspring: {len(queued)}

### New Population
| ID | Origin | Status |
|----|--------|--------|"""

        for ind in new_individuals:
            status = ind['status']
            if ind.get('fitness'):
                status = f"evaluated ({ind['fitness']['overall']:.1f})"
            changelog_entry += f"\n| {ind['id']} | {ind['origin']} | {status} |"

        append_changelog(changelog_entry)

        print(f"Generation {pop['generation']} created:")
        print(f"  {len(elites)} elites, {len(queued)} new candidates")
        for ind in new_individuals:
            f = ind['fitness']['overall'] if ind.get('fitness') else 'pending'
            print(f"  {ind['id']}: {ind['origin']} → {f}")

    def run(self, generations: int = 3, evaluate: bool = True, evolve: bool = True):
        """Run the full GA loop."""
        print(f"\n{'='*60}")
        print(f"TRELLIS.2 Genetic Algorithm Optimizer")
        print(f"Generations: {generations}")
        print(f"Examples: {self.example_ids}")
        print(f"{'='*60}\n")

        # Load pipeline and examples
        if evaluate:
            self.load_pipeline()
        examples = self.load_examples() if evaluate else {}

        for gen in range(generations):
            print(f"\n{'='*60}")
            print(f"GENERATION {gen + 1}/{generations}")
            print(f"{'='*60}")

            # Step 1: Evaluate queued individuals
            if evaluate and examples:
                self.evaluate_queued(examples)

            # Step 2: Evolve to create next generation
            if evolve:
                self.evolve()

            # Periodic GPU cleanup
            if evaluate:
                torch.cuda.empty_cache()

        # Final summary
        pop = load_population()
        evaluated = [ind for ind in pop['individuals'] if ind.get('fitness')]
        evaluated.sort(key=lambda x: x['fitness']['overall'], reverse=True)

        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total generations: {pop['generation']}")
        if evaluated:
            print(f"Best individual: {evaluated[0]['id']}")
            print(f"Best fitness: {evaluated[0]['fitness']['overall']:.1f}")
            print(f"\nTop 5 configurations:")
            for ind in evaluated[:5]:
                print(f"  {ind['id']}: {ind['fitness']['overall']:.1f}")
                for k, v in ind['params'].items():
                    if v != pop['fixed_params'].get(k) and v != pop['individuals'][0]['params'].get(k):
                        print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Genetic Algorithm Optimizer')
    parser.add_argument('--generations', type=int, default=3,
                        help='Number of generations to run')
    parser.add_argument('--population-size', type=int, default=8,
                        help='Population size (updates population.json)')
    parser.add_argument('--examples', type=str, default='4,6,7',
                        help='Comma-separated example IDs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate queued individuals, do not evolve')
    parser.add_argument('--evolve-only', action='store_true',
                        help='Only evolve (create next gen), do not evaluate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    example_ids = [int(x.strip()) for x in args.examples.split(',')]

    # Update population size if specified
    pop = load_population()
    if args.population_size != pop['population_size']:
        pop['population_size'] = args.population_size
        save_population(pop)

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(TRELLIS_ROOT, 'optimization', f'ga_run_{timestamp}')
    else:
        output_dir = args.output

    optimizer = GeneticOptimizer(
        output_dir=output_dir,
        example_ids=example_ids,
    )

    do_evaluate = not args.evolve_only
    do_evolve = not args.evaluate_only

    # evolve-only should always be exactly 1 generation step
    generations = 1 if args.evolve_only else args.generations

    optimizer.run(
        generations=generations,
        evaluate=do_evaluate,
        evolve=do_evolve,
    )


if __name__ == '__main__':
    main()
