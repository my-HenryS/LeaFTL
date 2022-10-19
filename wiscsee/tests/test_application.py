import unittest
import collections
import shutil
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.distributions

import config
from workflow import *
import wiscsim
from utilities import utils
from wiscsim.hostevent import Event, ControlEvent
from config_helper import rule_parameter
from pyreuse.helpers import shcmd
from config_helper import experiment
from wiscsim.learnedftl import *
from wiscsim.lsm_tree.bloom_filter import BloomFilter
from wiscsim.utils import *

def create_config(ftl_type="dftldes"):
    if ftl_type == "dftldes" or ftl_type == "learnedftl":
        conf = wiscsim.dftldes.Config()
    else:
        conf = wiscsim.nkftl2.Config()

    conf['ftl_type'] = ftl_type
    conf['SSDFramework']['ncq_depth'] = 1

    conf['flash_config']['n_pages_per_block'] = 256
    conf['flash_config']['n_blocks_per_plane'] = 2048
    conf['flash_config']['n_planes_per_chip'] = 4
    conf['flash_config']['n_chips_per_package'] = 1
    conf['flash_config']['n_packages_per_channel'] = 1
    conf['flash_config']['n_channels_per_dev'] = 16

    # set ftl
    conf['do_not_check_gc_setting'] = True
    conf.GC_high_threshold_ratio = 0.96
    conf.GC_low_threshold_ratio = 0

    conf['enable_simulation'] = True

    utils.set_exp_metadata(conf, save_data = False,
            expname = 'test_expname',
            subexpname = 'test_subexpname')

    conf['simulator_class'] = 'SimulatorDESNew'
    conf['trace_path'] = "/home/js39/datasets/MSR-Cambridge/prxy_1.csv"

    if ftl_type == "dftldes":
        logicsize_mb = 16
        conf.n_cache_entries = conf.n_mapping_entries_per_page * 256
        #conf.set_flash_num_blocks_by_bytes(int(logicsize_mb * 2**20 * 1.28))

    utils.runtime_update(conf)

    return conf

def parse_events(filename, lineno=float('inf'), format="MSR"):
    if "rocksdb" in filename:
        format = "blktrace"
    if "systor17" in filename:
        format = "systor"

    log_msg("parsing %s with %s format" % (filename, format))
    events = []
    # Dict<Format, Tuple<size_scale, time_scale, delimeter>>
    format_config = {"MSR" : (1, 100, ","), "blktrace" : (512, 1000**3, " "), "systor" : (1, 1000**3, ",")} 
    size_scale = format_config[format][0]
    time_scale = format_config[format][1]
    delimeter = format_config[format][2]

    with open(filename) as fp:
        t_start = None
        last_t = 0
        active_events = 0
        for i, raw in enumerate(fp):
            # parse trace
            line = raw.strip().split(delimeter)
            if format == "MSR":
                t, p, d, mode, offset, size, t0 = line
                t, d, offset, size, t0 = int(t), int(d), int(offset), int(size), int(t0)
            elif format == "blktrace":
                line = filter(lambda _: _ != '', line)
                raise NotImplementedError
            elif format == "systor":
                if i == 0:
                    continue
                t, t0, mode, d, offset, size = line
                if t0 == "":
                    t0 = 0.0
                t, d, offset, size, t0 = float(t), int(d), int(offset), int(size), float(t0)

            # shift timestamp
            if not t_start:
                t_start = t
            t -= t_start

            # scale trace
            offset *= size_scale
            size *= size_scale
            t = int(t*time_scale)
            if mode in ["Read", "R"]:
                op = OP_READ
            elif mode in ["Write", "W"]:
                op = OP_WRITE

            # create event
            if t < last_t:
                continue
            events += [ControlEvent(OP_SLEEP, arg1=t - last_t)]
            events += [Event(512, 0, op, offset, size, timestamp=t)]
            active_events += 1
            last_t = t
        
            # termination
            if i > lineno:
                break
    
    # timestamp from traces might not be sorted! (now we abort the unsorted ones)
    # events = sorted(events, key=lambda event: event.timestamp)
    # for i in range(0, len(events)):
    #     events.insert(i*2, ControlEvent(OP_SLEEP, arg1=None))
    # last_t = 0
    # for i in range(0, len(events), 2):
    #     sleep, event = events[i], events[i+1] 
    #     sleep.arg1 = event.timestamp - last_t
    #     last_t = event.timestamp

    log_msg("Total active events %d" % active_events)
    return events

class TestApp(unittest.TestCase):
    def test_segment(self):
        plr = PLR(gamma=16)
        points = [(100,0),(101,1),(102,2),(103,3),(104,4),(120,4),(121,5),(122,6),(123,7),(124,8)]
        # points = [(701, 41943840), (702, 41943841), (781, 41943842), (782, 41943843), (855, 41943844), (856, 41943845)]
        # points = [(0,64), (1, 65), (31, 66), (35, 67)]
        plr.learn(points)
        seg = plr.segments[0]
        print(plr.segments)
        # plr.init()
        # points = [(100,0),(101,1),(102,2),(103,3),(104,4)]
        # points = [(781, 8388684), (782, 8388685)]
        # plr.learn(points)
        # seg2 = plr.segments[0]

        # print(seg, seg2)
        # print(Segment.merge(seg,seg2))
        #print(Segment.merge(seg2,seg))

    def test_plr(self):
        plr = PLR(gamma=4)
        points = [(3314206, 150996211), (3314210, 150996212), (3314211, 150996213), (3314212, 150996214), (3314213, 150996215), (3314215, 150996216), (3314216, 150996217), (3314217, 150996218), (3314218, 150996219), (3314219, 150996220), (3314220, 150996221), (3314221, 150996222), (3314222, 150996223)]#,(103,3),(120,4),(121,5),(122,6),(123,7),(124,8)]
        wrong_points = [(104,None),(105,None)]
        # points = [[x,x+10] for x in range(100)]
        # points += [[x,x+1] for x in range(100,200)]
        plr.learn(points)
        seg = plr.segments[0]
        seg.x1 = 3314219
        for point in points:
            print("prediction: %s, actual: %s" % (plr.segments[0].get_y(point[0]), point[1]))
        for point in wrong_points:
            print("prediction: %s, actual: %s" % (plr.segments[0].get_y(point[0]), point[1]))

        print(seg, seg.filter, seg.memory)

    def test_logplr(self):
        plr = LogPLR(gamma=3)
        # points = [(116292, 250), (116293, 251), (10000000, 252)]
        # plr.update(points)
        # points = [[x,x+10] for x in range(100)]
        # plr.update(points, 0)
        # points = [[x,x+1] for x in range(100,200)]
        # plr.update(points, 0)
        # points = [[x,x+100] for x in range(0,11)]
        # plr.update(points, 0)
        # points = [[x,x+100] for x in range(1,10)]
        # plr.update(points, 0)
        # points = [[x,x+200] for x in range(80,300)]
        # plr.update(points, 0)
        # points = [(100,0),(101,1),(102,2),(103,3),(120,4),(121,5),(122,6),(123,7),(124,8)]
        # plr.update(points, 0)
        points = [(781, 8388684), (782, 8388685)]
        plr.update(points, 0)
        points = [(701, 41943840), (702, 41943841), (781, 41943842), (782, 41943843), (855, 41943844), (856, 41943845)]
        plr.update(points, 0)
        print(plr)

    def test_learnedftl(self):
        gamma, lineno = 1e-4, 10000
        print("gamma = %d, lineno = %d" % (gamma, lineno))

        conf = create_config('learnedftl')
        conf['gamma'] = gamma
        conf['cache_size'] = 16
        conf['dry_run'] = False
        conf['trace_path'] = "/home/js39/datasets/MSR-Cambridge/src1_0.csv" #"/home/js39/datasets/virtual/systor17-01/2016022209-LUN1.csv"
        events = parse_events(conf['trace_path'], lineno=lineno)
        wf = Workflow(conf)
        sim = wf.run_simulator(events)
        mapping_table = sim.ssd.ftl.metadata.mapping_table
        reference_mapping_table = sim.ssd.ftl.metadata.reference_mapping_table
        # print(mapping_table.runs[-10:])
        log_msg(sim.ssd.ftl.hist)
       
        log_msg("# of levels: %d" % len(mapping_table.runs))
        log_msg("# of segments: %d" % len(mapping_table.segments))
        log_msg("estimated learnedftl memory footprint: %d B" % mapping_table.memory)
        log_msg("estimated dftl memory footprint: %d B" % (len(reference_mapping_table.mapping_table)*8))
        
        for lpn in reference_mapping_table.mapping_table:
            if len(list(mapping_table.lookup(lpn))) >= 3:
                log_msg("lpn:", lpn)
                for ppn, accurate, seg in mapping_table.lookup(lpn):
                    log_msg("learned segment:", seg.full_str())
        

    def test_hybridftl(self):
        conf = create_config('nkftl2')

        ctrl_event = ControlEvent(OP_ENABLE_RECORDER)
        events = parse_events(conf['trace_path'], lineno=1000000)
        #events += [Event(512, 0, OP_WRITE, i*4096, 4096) for i in reversed(range(257))]

        wf = Workflow(conf)
        sim = wf.run_simulator([ctrl_event]+events)

        pftl = [log_group._page_map \
                for dgn, log_group in sim.ssd.ftl.log_mapping_table.log_group_info.items() ]
        len_pftl = sum([len(mapping)  for mapping in pftl])

        bftl = sim.ssd.ftl.data_block_mapping_table.logical_to_physical_block
        len_bftl = len(bftl)

        print("pftl: %d " % len_pftl)
        print("bftl: %d " % len_bftl)

    def test_dftl(self):
        lineno = 1000000
        print("lineno %d" % (lineno))

        conf = create_config("dftldes")

        ctrl_events = [ControlEvent(OP_DISABLE_RECORDER)]
        events = parse_events(conf['trace_path'], lineno=1000000)

        #events += [Event(512, 0, OP_WRITE, i*4096, 4096) for i in reversed(range(257))]


        wf = Workflow(conf)
        sim = wf.run_simulator(ctrl_events+events)
        # dftl print
        len_dftl = len([row for row in sim.ssd.ftl._mappings._lpn_table.rows() if row.state=="USED"])

        print("dftl: %d " % len_dftl)

    def test_bloom(self):
        bf = BloomFilter(256, 0.1)
        print(bf.bit_array_size, bf.num_hash_fns)
        bf.add(str(1))
        bf.add(str(40))
        bf.add(str(100))
        for i in range(100):
            print(i, bf.check(str(i)))
    
    def test_pvb(self):
        conf = create_config()
        pvb = PageValidityBitmap(conf)
        pvb.validate_block(0)
        pvb.invalidate_page(100)
        pvb.validate_page(10000)
        assert(pvb.is_page_valid(10000) == True)
        assert(pvb.is_page_invalid(100) == True)
        assert(pvb.is_page_invalid(1000) == True)
        
    def test_plot(self):
        results = []
        tols = [0, 4, 8, 16]
        colors = ['blue', 'purple', 'green','orange']
        lines = ['-', '--', '-.','dotted']
        for tol in tols:
            conf = create_config()
            conf['ftl_type'] = 'learnedftl'
            conf['tolerance'] = tol
            events = parse_events(conf['trace_path'], lineno=60000)
            wf = Workflow(conf)
            sim = wf.run_simulator(events)
            print("# of segments: %d" % len(sim.ssd.ftl.mapping_table.segments))
            results.append(sorted(sim.ssd.ftl.buffer.dist))

        fig, ax = plt.subplots()
        for i, dist in enumerate(results):
            print("dist: %d %d" % (len(dist), sum(dist)))
            y, x = np.histogram(dist, bins=[2**_ for _ in range(13)])
            print(tols[i], y, x)
            # ecdf = statsmodels.distributions.ECDF(dist)
            # Plot cdf
            ax.plot(x[:-1], y, label=("tolerance=%d" % tols[i]), color=colors[i], linestyle = lines[i])
            ax.set_xlabel('Length of segments')
            ax.set_xlim([1,4096])
            ax.set_xscale('log', basex=2)
            ax.set_ylabel('% of segments')
            # ax.set_yscale('log', basey=2)
            # ax.set_ylim([0,1])
        plt.grid()
        plt.legend()
        fig.savefig("/home/js39/software/wiscsee/results/hist-600000.pdf")

        # # Plot hist
        # ax.hist(dist, bins=512)
        # ax.set_xlabel('Consecutive writes')
        # ax.set_ylabel('#')
        # fig.savefig("/home/js39/software/wiscsee/results/cdf-4.pdf")


if __name__ == '__main__':
    unittest.main()

