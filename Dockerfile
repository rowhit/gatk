# Using OpenJDK 8
FROM broadinstitute/gatk:gatkbase-1.2

ADD . /gatk

WORKDIR /gatk
RUN /gatk/gradlew clean compileTestJava installAll localJar

WORKDIR /root

# Make sure we can see a help message
RUN ln -sFv /gatk/build/libs/gatk.jar
RUN java -jar gatk.jar -h

#Setup test data
WORKDIR /gatk
# Create link to where test data is expected
RUN ln -s /testdata src/test/resources

# Create a simple unit test runner
ENV CI true
RUN echo "cd /gatk/ && ./gradlew jacocoTestReport" >/root/run_unit_tests.sh

WORKDIR /root
RUN cp -r /root/run_unit_tests.sh /gatk
RUN cp -r gatk.jar /gatk
RUN cp -r install_R_packages.R /gatk

#Start python environment
RUN	apt-get install -y wget libglib2.0-0

ENV ANACONDA_URL https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
ENV ANACONDA_PATH /opt/anaconda
ENV PATH $ANACONDA_PATH/bin:$PATH

ENV DOWNLOAD_DIR /downloads
RUN mkdir $DOWNLOAD_DIR

# For now get full anaconda pkg. Eventually this should move to gatkBase.
# TODO: Verify dc13fe5502cd78dd03e8a727bb9be63f
RUN wget -nv -O $DOWNLOAD_DIR/anaconda.sh $ANACONDA_URL && \
	/bin/bash $DOWNLOAD_DIR/anaconda.sh -b -p $ANACONDA_PATH && \
	conda update -y conda && \
	conda update -y numpy && \
	conda update -y scipy && \
	conda update -y pandas && \
	rm $DOWNLOAD_DIR/anaconda.sh && \
	conda clean -yt

RUN conda info -a
# Need a test here that verifies the actual env
RUN ipython -c "print 'GATK Python Env done'"

#End Python environment

WORKDIR /gatk