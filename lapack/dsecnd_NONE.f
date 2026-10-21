*> \brief \b DSECND returns nothing
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*      DOUBLE PRECISION FUNCTION DSECND( )
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>  DSECND returns nothing instead of returning the user time for a process in seconds.
*>  If you are using that routine, it means that neither EXTERNAL ETIME,
*>  EXTERNAL ETIME_, INTERNAL ETIME, INTERNAL CPU_TIME is available  on
*>  your machine.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2011
*
*> \ingroup auxOTHERauxiliary
*
*  =====================================================================
*> SPDX-FileCopyrightText: 1992-2015 The University of Tennessee and The University of Tennessee Research Foundation
*> SPDX-FileCopyrightText: 2000-2015 The University of California Berkeley
*> SPDX-FileCopyrightText: 2006-2015 The University of Colorado Denver
*> SPDX-License-Identifier: BSD-3-Clause-Open-MPI
      DOUBLE PRECISION FUNCTION DSECND( )
*
*  -- LAPACK auxiliary routine (version 3.4.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
* =====================================================================
*
      DSECND = 0.0D+0
      RETURN
*
*     End of DSECND
*
      END
