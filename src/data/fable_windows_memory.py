def limit_current_process_committed_memory(limit_bytes: int):
    import win32api
    import win32job

    if type(limit_bytes) is not int or limit_bytes <= 0:
        raise ValueError('positive integer committed-memory limit required')
    job = win32job.CreateJobObject(None, '')
    information = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    information['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
    information['ProcessMemoryLimit'] = limit_bytes
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, information)
    win32job.AssignProcessToJobObject(job, win32api.GetCurrentProcess())
    observed = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    if observed['ProcessMemoryLimit'] != limit_bytes:
        raise RuntimeError('Windows did not apply requested memory limit')
    return job
