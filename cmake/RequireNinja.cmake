function(kfcore_require_ninja output_variable)
  unset(_kfcore_ninja_executable)
  unset(_kfcore_ninja_executable CACHE)
  find_program(_kfcore_ninja_executable NAMES ninja ninja.exe)
  if(NOT _kfcore_ninja_executable)
    message(FATAL_ERROR
      "TensorRT runtime installed-consumer test requires Ninja (ninja or ninja.exe). "
      "Install Ninja and add it to PATH before configuring KFCore.")
  endif()

  set(${output_variable} "${_kfcore_ninja_executable}" PARENT_SCOPE)
  unset(_kfcore_ninja_executable CACHE)
endfunction()
