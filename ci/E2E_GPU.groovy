int total_timeout_minutes = 60*2
def knowhere_wheel=''
pipeline {
    agent {
        kubernetes {
            inheritFrom 'default'
            yamlFile 'ci/pod/e2e.yaml'
            defaultContainer 'main'
        }
    }

    options {
        timeout(time: total_timeout_minutes, unit: 'MINUTES')
        buildDiscarder logRotator(artifactDaysToKeepStr: '30')
        parallelsAlwaysFailFast()
        disableConcurrentBuilds(abortPrevious: true)
        preserveStashes(buildCount: 10)
    }
    stages {
        stage("Build"){

            steps {
                container("build"){
                    script{
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: "echo ${env.GIT_COMMIT} | cut -b 1-7 ").trim()
                        version="${env.CHANGE_ID}.${date}.${gitShortCommit}"
                        sh "apt-get update || true"
                        sh "apt-get install dirmngr -y"
                        sh "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA"
                        sh "apt-get install build-essential libopenblas-dev ninja-build git -y"
                        sh "git config --global --add safe.directory '*'"
                        sh "git submodule update --recursive --init"
                        sh "mkdir build"
                        sh "cd build/ && cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWITH_UT=ON -DWITH_RAFT=ON"
                        sh "pwd"
                        sh "ls -la"
                        sh "cd build/ && make -j"
                        sh "cd .."
                        sh "cd python  && VERSION=${version} python3 setup.py bdist_wheel"
                        dir('python/dist'){
                        knowhere_wheel=sh(returnStdout: true, script: 'ls | grep .whl').trim()
                        archiveArtifacts artifacts: "${knowhere_wheel}", followSymlinks: false
                        }
                        // stash knowhere info for rebuild E2E Test only
                        sh "echo ${knowhere_wheel} > knowhere.txt"
                        stash includes: 'knowhere.txt', name: 'knowhereWheel'
                    }
                }
            }
        }
        stage("Test"){
            steps {
                script{
                    if ("${knowhere_wheel}"==''){
                        dir ("knowhereWheel"){
                            try{
                                unstash 'knowhereWheel'
                                knowhere_wheel=sh(returnStdout: true, script: 'cat knowhere.txt | tr -d \'\n\r\'')
                            }catch(e){
                                error "No knowhereWheel info remained ,please rerun build to build new package."
                            }
                        }
                    }
//                  checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [],
                    checkout([$class: 'GitSCM', branches: [[name: '*/feature-2.0']], extensions: [],
                    userRemoteConfigs: [[credentialsId: 'milvus-ci', url: 'https://github.com/milvus-io/knowhere-test.git']]])
                    dir('tests'){
                      unarchive mapping: ["${knowhere_wheel}": "${knowhere_wheel}"]
                      sh "ls -lah"
                      sh "apt update"
                      sh "apt install python3-pip"
                      sh "nvidia-smi"
                      sh "pip3 install ${knowhere_wheel} \
                          && pip3 install -r requirements.txt --timeout 30 --retries 6  && pytest -v -m 'L0 and gpu'"
                    }
                }
            }
            post{
                always {
                    script{
                        sh 'cp /tmp/knowhere_ci.log knowhere_ci.log'
                        archiveArtifacts artifacts: 'knowhere_ci.log', followSymlinks: false
                    }
                }
            }
        }

    }
}
