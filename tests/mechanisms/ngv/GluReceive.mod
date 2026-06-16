COMMENT
/**
 * @file GlutReceive.mod
 * @brief
 * @author keller
 * @date 2011-08-17
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice
 */
ENDCOMMENT

TITLE Glutamate Receive Object


COMMENT
make a glutamate receive function
ENDCOMMENT


NEURON {
    THREADSAFE
    POINT_PROCESS GlutReceive
    RANGE glut
}


PARAMETER {
    glut
}


ASSIGNED {
}


STATE {
}


INITIAL{
    glut=0
}


BREAKPOINT {

}


PROCEDURE state() {
}


NET_RECEIVE (weight){
    INITIAL{
    }
    glut=glut+1
:VERBATIM
:        printf("RECEIVE\n");
:ENDVERBATIM

}

