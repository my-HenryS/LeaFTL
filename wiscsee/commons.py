# All the constants here are meant to be used by
# from commons import *

PAGE_PROGRAMMED, PAGE_ERASED = 'PAGE_PROGRAMMED', 'PAGE_ERASED'

BYTE, KB, MB, GB, TB = [2**(10*i) for i in range(5)]

MILLION = 10**6

# unit is based on nanoseconds
SEC, MILISEC, MICROSEC, NANOSEC = [ 1000**3, 1000**2, 1000, 1 ]
# SEC, MILISEC, MICROSEC, NANOSEC = [ 1.0, 0.001, 0.000001, 0.000000001 ]

OP_READ, OP_WRITE, OP_ERASE, OP_SLEEP = 'OP_READ', 'OP_WRITE', 'OP_ERASE', 'OP_SLEEP'
OP_DISCARD = 'OP_DISCARD'
OP_REC_TIMESTAMP = 'OP_REC_TIMESTAMP'
OP_BARRIER = 'OP_BARRIER'
OP_CLEAN = 'OP_CLEAN'
OP_ENABLE_RECORDER = 'OP_ENABLE_RECORDER'
OP_DISABLE_RECORDER = 'OP_DISABLE_RECORDER'
OP_SHUT_SSD = 'OP_SHUT_SSD'
OP_END_SSD_PROCESS = 'OP_END_SSD_PROCESS'
OP_WORKLOADSTART = 'OP_WORKLOADSTART'
OP_DROPCACHE = 'OP_DROPCACHE'
OP_FALLOCATE = 'OP_FALLOCATE'
OP_ARG_KEEPSIZE = 'OP_ARG_KEEPSIZE'
OP_ARG_NOTKEEPSIZE = 'OP_ARG_NOTKEEPSIZE'
OP_FSYNC = 'OP_FSYNC'
OP_FDATASYNC = 'OP_FDATASYNC'
OP_FLUSH_TRANS_CACHE = 'OP_FLUSH_TRANS_CACHE'
OP_DROP_TRANS_CACHE = 'OP_DROP_TRANS_CACHE'
OP_PURGE_TRANS_CACHE = 'OP_PURGE_TRANS_CACHE'
OP_NOOP = 'OP_NOOP'
OP_CALC_GC_DURATION = 'OP_CALC_GC_DURATION'
OP_REC_FLASH_OP_CNT = 'OP_REC_FLASH_OP_CNT'
OP_REC_FOREGROUND_OP_CNT = 'OP_REC_FOREGROUND_OP_CNT'
OP_REC_CACHE_HITMISS = 'OP_REC_CACHE_HITMISS'
OP_NON_MERGE_CLEAN = 'OP_NON_MERGE_CLEAN'
OP_CALC_NON_MERGE_GC_DURATION = 'OP_CALC_NON_MERGE_GC_DURATION'
OP_REC_BW = 'OP_REC_BW'

TAG_BACKGROUND = "BACKGROUND"
TAG_FOREGROUND = "FOREGROUND"

TESTALL = False

F2FS_IPU_DISABLE = 0
F2FS_IPU_FORCE = 1 << 0
F2FS_IPU_SSR = 1 << 1
F2FS_IPU_UTIL = 1 << 2
F2FS_IPU_SSR_UTIL = 1 << 3
F2FS_IPU_FSYNC = 1 << 4

