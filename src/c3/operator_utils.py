import contextlib
import logging
import os

# converts string in form [cos|s3]://access_key_id:secret_access_key@endpoint/bucket/path to
# access_key_id, secret_access_key, endpoint, path - path includes bucket name
def explode_connection_string(cs):
    if cs is None:
        return None
    if cs.startswith('cos') or cs.startswith('s3'):
        buffer=cs.split('://')[1]
        access_key_id=buffer.split('@')[0].split(':')[0]
        secret_access_key=buffer.split('@')[0].split(':')[1]
        endpoint=f"https://{buffer.split('@')[1].split('/')[0]}"
        path='/'.join(buffer.split('@')[1].split('/')[1:])
        return (access_key_id, secret_access_key, endpoint, path)
    else:
        return (None, None, None, cs)
        # TODO consider cs as secret and grab connection string from kubernetes
    

def run_and_log(cos_conn, log_folder, task_id, command_array):
    log_root_name = time.time()
    job_id = ('-').join(command_array).replace('/','-') # TODO get a unique job id
    job_id = re.sub(r'[^a-zA-Z0-9]', '-', job_id)
    task_id = re.sub(r'[^a-zA-Z0-9]', '-', task_id)
    std_out_log_name = f'{job_id}-{task_id}-{log_root_name}-stdout.log'
    std_err_log_name = f'{job_id}-{task_id}-{log_root_name}-stderr.log'
    with open(std_out_log_name,'w') as so:
        with open(std_err_log_name,'w') as se:
            with contextlib.redirect_stdout(so):
                with contextlib.redirect_stderr(se):
                    logging.info('-----INVOKING TASK-----------------------------------')
                    logging.info(f'Task ID: {task_id}')
                    logging.info(f'Command: {command_array}')    
                    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
                    output = result.stdout.decode('utf-8')
                    logging.info("Output:", output)
                    logging.info("Return code:", result.returncode)
    cos_conn.put(std_out_log_name,os.path.join(log_folder,std_out_log_name))
    cos_conn.put(std_err_log_name,os.path.join(log_folder,std_err_log_name))
    os.remove(std_out_log_name)
    os.remove(std_err_log_name)