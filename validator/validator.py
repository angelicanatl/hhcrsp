#!/usr/bin/env python3
import click
from solution import Solution, validate_solution
from instance import Instance
import sys
import json
import pandas as pd
import os
import glob
from pathlib import Path
import shutil
import logging
log = logging.getLogger('generator')
logging.basicConfig()

@click.group()
@click.option('-v', '--verbose', default=False)
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

@cli.command()
@click.argument('instances', type=click.Path(readable=True, dir_okay=True, file_okay=True, path_type=Path))
@click.argument('table', type=click.Path(writable=True, file_okay=True, path_type=Path))
@click.option('--suffix', type=str, default='txt')
@click.pass_context
def features(ctx, instances, table, suffix):    
    if instances.is_dir():
        result = pd.DataFrame()
        for filename in glob.glob(os.path.join(instances, f"*.{suffix}")):
            with open(filename) as f:
                inst = Instance(f.read())
                tmp = pd.json_normalize([inst.compute_features()])
            tmp['instance'] = os.path.basename(filename)
            tmp.set_index('instance', inplace=True)
            result = pd.concat([result, tmp])
        if result.empty:
            log.info(f"No instance found in {instances}")
            sys.exit(0)
    else:
        result = pd.DataFrame()
        with open(instances) as f:
            inst = Instance(f.read())
            tmp = pd.json_normalize([inst.compute_features()])
            tmp['instance'] = instances.name
            result = pd.concat([result, tmp])
    if table.suffix.startswith('.xls'):
        result.to_excel(table)
    elif table.suffix.startswith('.csv'):
        result.to_csv(table)
    else:
        result.to_string(table)


@cli.command()
@click.argument('instance', type=click.File('r'))
@click.pass_context
def check_instance(ctx, instance):
    inst = Instance(instance.read())
    if ctx.obj.get('VERBOSE'):
        pass
    else:
        print("Done")

@cli.command()
@click.argument('instance', type=click.File('r'))
@click.argument('solution', type=click.File('r'))
@click.option('--scale-integers', type=int, default=1)
@click.option('-rt', '--route-tardiness', is_flag=True, show_default=True, default=False, help="Compute also route tardiness at depot.")
@click.pass_context
def check_solution(ctx, instance, solution, scale_integers, route_tardiness):
    inst = Instance(instance.read())
    sol = Solution(solution.read(), inst, scale_integers)
    cost = validate_solution(inst, sol, route_tardiness)
    if ctx.obj.get('VERBOSE'):
        print("Routes: ", sol.routes)
        print("Patients: ", sol.served_patients)
        print("Cost: ", cost)
    else:
        try:
            print(json.dumps(cost))
        except Exception as e:
            print(e, cost)

@cli.command()
@click.argument('instance', type=click.File('r'))
@click.argument('solution', type=click.File('r'))
@click.option('-f', '--format', type=click.Choice(['kummer', 'el'], case_sensitive=False), required=True)
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.pass_context
def convert_solution(ctx, instance, solution, format, output):
    inst = Instance(instance.read())
    sol = Solution(solution.read(), inst)
    if format == 'kummer':
        converted = sol.to_kummer()        
    else:
        sys.exit(f"Conversion to format {format} not supported yet")
    if output:
        with open(output, 'w') as f:
            f.write(converted)
    else:
        print(converted)

@cli.command()
@click.argument('instance', type=click.File('r'))
@click.option('-o', '--output', type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.pass_context
def convert_instance(ctx, instance, output):
    inst = Instance(instance.read())
    converted = inst.to_json()        
    if output:
        with open(output, 'w') as f:
            json.dump(converted, f, indent=4)
    else:
        print(json.dumps(converted, indent=4))

@cli.command()
@click.argument('instance', type=click.File('r'))
@click.argument('solution', type=click.File('r'))
@click.option('-s', '--scale', type=float, default=1.0)
@click.pass_context
def show_graph(ctx, instance, solution, scale):
    inst = Instance(instance.read())
    sol = Solution(solution.read(), inst)
    sol.to_graph(inst, scale)

@cli.command()
@click.argument('instance_dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True))
@click.argument('solution_dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True))
@click.argument('best_dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.pass_context
def extract_best(ctx, instance_dir, solution_dir, best_dir):
    if not best_dir.exists():
        best_dir.mkdir(parents=True)
    df = pd.DataFrame()
    for instance_name in instance_dir.iterdir():
        pattern = f'{instance_name.stem}'
        with open(instance_name) as f:
            inst = Instance(f.read())
        for solution_name in solution_dir.glob(f'{pattern}*'):
            with open(solution_name) as f:
                try:
                    sol = Solution(f.read(), inst)
                    result = validate_solution(inst, sol)
                    del result['tardiness']
                    del result['route_tardiness']
                    result = { 'instance': instance_name.name } | result | { 'solution': solution_name }
                    df = pd.concat([df, pd.DataFrame([ result  ])])
                except AssertionError:
                    continue
    df = df.sort_values('total_cost').drop_duplicates('instance')    
    for index, data in df.iterrows():
        solution = Path(data['solution'])
        shutil.copy(solution, best_dir)
    df['solution'] = df['solution'].map(lambda s: Path(s).name)
    df.reset_index().drop(columns='index').sort_values('instance').set_index('instance').to_markdown('best.md')




if __name__ == "__main__":
    cli(obj={})