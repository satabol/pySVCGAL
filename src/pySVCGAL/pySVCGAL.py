import ctypes
import numpy as np
import sys
import traceback
import itertools

#print("== pySVCGAL.py start =================================================")

from pySVCGAL.clib import load_library
SVCGAL_clib = load_library.load_library()

class MESH_DATA(ctypes.Structure):
    _fields_ = [
        ("has_error", ctypes.c_bool, ),
        ("str_error", ctypes.c_char_p, ),

        ("polygon_id", ctypes.c_int),
        
        # if has_error==False
        # results meshes
        ("nn_verts", ctypes.c_int),
        ("nn_edges", ctypes.c_int),
        ("nn_faces", ctypes.c_int),
        ("vertices", ctypes.POINTER((ctypes.c_float)*3), ),
        ("edges", ctypes.POINTER( (ctypes.c_int)*2), ),
        ("faces", ctypes.POINTER( (ctypes.c_int)*3), ),
        
        # if has_error==False
        # what contours are wrong (for visualize debugging)
        ("ftcs_count"           , ctypes.c_int),
        ("ftcs_vertices_counter", ctypes.POINTER( (ctypes.c_int) ), ),
        ("ftcs_vertices_description", ctypes.POINTER( (ctypes.c_char_p) ), ),
        ("ftcs_vertices_list"   , ctypes.POINTER( (ctypes.c_float)*3 ), ),

    ]

def pySVCGAL_extrude_skeleton(
        object_id,      # object id - int number
        polygon_id,     # polygon id - int number
        height,         # float      - single float value
        vertices,       # list of list of vertices of contours. [[v1,v2,v3,...], [v4,v5,v6,v7,...], ...]
        angles,         # list of list of angles of edges of every contour (in angles, not radians) [[]]
        exclude_height, # calc max height if mesh
        only_tests_for_valid, # do not calc Skeleton, only tests
        verbose         # verbose debug output (True - more info / False - less info)
    ):

    """
    Documentation
    """   

    extrude_stright_skeleton = SVCGAL_clib.extrude_stright_skeleton
    extrude_stright_skeleton.argtypes = [
        ctypes.c_int,                       # contour_id
        ctypes.c_float,                     # height
        ctypes.c_int,                       # in_count_of_contours
        ctypes.POINTER((ctypes.c_int) ),    # in_lens_of_contours
        ctypes.c_int,                       # in_count_of_angles
        ctypes.POINTER((ctypes.c_int) ),    # in_lens_of_angles
        ctypes.POINTER((ctypes.c_float)*3), # vertices 
        ctypes.POINTER((ctypes.c_float)),   # angles per edge of one contour
        ctypes.c_bool,                      # exclude_height
        ctypes.c_bool,                      # only_tests_for_valid
        ctypes.c_bool,                      # verbose (additional info output to a console)
    ]
    extrude_stright_skeleton.restype = ctypes.POINTER(MESH_DATA)

    free_mem = SVCGAL_clib.free_mem
    free_mem.argtypes = [
        ctypes.POINTER(MESH_DATA)
    ]

    # Создать совмещённый список всех вершин:
    all_vertices            = list(itertools.chain(*vertices))
    all_angles              = list(itertools.chain(*angles))
    lens_of_contours        = [len(l) for l in vertices]
    in_count_of_contours    = len(lens_of_contours)
    lens_of_angles          = [len(l) for l in angles]
    in_count_of_angles      = len(lens_of_angles)

    in_lens_of_contours = ctypes.ARRAY(in_count_of_contours, ctypes.c_int)(*[(ctypes.c_int)(l) for l in lens_of_contours])
    in_lens_of_angles   = ctypes.ARRAY(  in_count_of_angles, ctypes.c_int)(*[(ctypes.c_int)(l) for l in lens_of_angles  ])

    in_vertices = ctypes.ARRAY( len(all_vertices), ctypes.c_float*3)(*[(ctypes.c_float*3)(*vert) for vert in all_vertices])
    in_angles   = ctypes.ARRAY( len(all_angles)  , ctypes.c_float  )(*[(ctypes.c_float  )(    a) for a in all_angles])
    
    md = None
    try:
        md = extrude_stright_skeleton(
            polygon_id,
            height,
            in_count_of_contours,
            in_lens_of_contours,
            in_count_of_angles,
            in_lens_of_angles,
            in_vertices,
            in_angles,
            only_tests_for_valid,
            exclude_height,
            verbose
        )
        ################ Get Results ################
        mdc = md.contents

        str_error = None
        if(mdc.str_error is not None):
            str_error = mdc.str_error.decode("ascii")
        new_mesh = {
            'object_id' : object_id,
            'polygon_id': mdc.polygon_id,
            'vertices'  : [],
            'edges'     : [],
            'faces'     : [],
            'has_error'                 : mdc.has_error,
            'str_error'                 : str_error,
            'ftcs_count'                : mdc.ftcs_count,
            'ftcs_vertices_counter'     : -1,
            'ftcs_vertices_description' : [],
            'ftcs_vertices_list'        : [],
        }

        if mdc.has_error==False:
            #### Extract New Vertices #### 
            #new_vertices = [ mdc.vertices[i][:] for i in range(mdc.nn_verts)]
            new_vertices = [ tuple(mdc.vertices[i]) for i in range(mdc.nn_verts)]

            #### Extract New Edges #### 
            new_edges = [ tuple(mdc.edges[i]) for i in range(mdc.nn_edges)]
            #### Extract New Faces #### 
            new_faces = [ tuple(mdc.faces[i]) for i in range(mdc.nn_faces)]
            
            free_mem(mdc)

            new_mesh = new_mesh | {
                'vertices'  : new_vertices,
                'edges'     : new_edges,
                'faces'     : new_faces,
            }
        else: # mdc.has_error==True:
            try:
                # Общие данные:

                if mdc.ftcs_count>0:
                    # This is error of triangulation. Write these contours into returned data:
                    new_mesh_ftcs_vertices_counter = [ mdc.ftcs_vertices_counter[i] for i in range( mdc.ftcs_count) ]
                    new_mesh_ftcs_vertices_description = [ mdc.ftcs_vertices_description[i] for i in range( mdc.ftcs_count) ]
                    new_mesh_ftcs_vertices_list = []

                    idx=0
                    for I in range(mdc.ftcs_count):
                        failed_contour_length = new_mesh_ftcs_vertices_counter[I]
                        failed_contour_vertices = [ tuple(mdc.ftcs_vertices_list[i+idx]) for i in range(failed_contour_length) ]
                        new_mesh_ftcs_vertices_list.append(failed_contour_vertices)
                        idx+=failed_contour_length
                    
                    new_mesh = new_mesh | {
                        'ftcs_vertices_counter'     : new_mesh_ftcs_vertices_counter,
                        'ftcs_vertices_description' : new_mesh_ftcs_vertices_description,
                        'ftcs_vertices_list'        : new_mesh_ftcs_vertices_list,
                    }
                else:
                    pass

                free_mem(mdc)
                pass
            except Exception as ex:
                new_mesh = new_mesh | {
                    # 'polygon_id'        : polygon_id,
                    # 'vertices'          : [],
                    # 'faces'             : [],
                    'has_error'         : True,
                    'str_error'         : ex
                }
                pass

        return new_mesh

    except Exception as _ex:
        str_error = ""
        if md is not None:
            mdc = md.contents
            mdc_str_error = ""
            if mdc.str_error is not None:
                mdc_str_error = mdc.str_error.decode("ascii")
                str_error = f"Unexpected exception while calculating data: {mdc_str_error}."
            else:
                str_error = f"Unexpected exception while calculating data. No internal addition info."
            free_mem(mdc)
        else:
            str_error = "General unexpected exception."
        str_error_summ = f"{str_error} polygon_id: {polygon_id}, {_ex}." 
        #raise Exception()
        new_mesh = {
            'polygon_id': polygon_id,
            'vertices'  : [],
            'faces'     : [],
            'has_error'                 : True,
            'str_error'                 : str_error_summ,
            'ftcs_count'                : 1,
            'ftcs_vertices_counter'     : len(all_vertices),
            'ftcs_vertices_description' : [],
            'ftcs_vertices_list'        : [all_vertices],
        }
        return new_mesh

if __name__ == "__main__":
    pass