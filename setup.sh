#!bash/sh

git submodule update --init --recursive # to get onnx protobuf files

#RUNTIME_VERSION=$(dotnet --list-runtimes | grep NETCore | sed 's/[^ ]* \([^ ]*\).*/\1/' | sort -Vr | head -n 1)
#DOTNET_CORE=$(dirname $(which dotnet))/shared/Microsoft.NETCore.App/$RUNTIME_VERSION
#DOTNET_SDK=$(dirname $(which dotnet))/sdk/$(dotnet --version)

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# TODO fix up hardcoding...
RUNTIME_VERSION=6.0.404
DOTNET_SDK=~/.dotnet/sdk/$RUNTIME_VERSION
DOTNET_CORE=~/.dotnet/shared/Microsoft.NETCore.App/6.0.12
SCRIPT_DIR=~/EE/Git/FSharp.ML.Onnx/

CSC=$DOTNET_SDK/Roslyn/bincore/csc.dll


pip3 install onnx # version 1.14


#protos=("onnx-ml" "onnx-operators-ml" "onnx-operators" "onnx")

protos=("onnx-ml")

DM_OUT=tmp
mkdir -p $DM_OUT

for str in ${protos[@]}; do
  protoc --csharp_out=$DM_OUT/ -Isubmodules/onnx onnx/$str.proto3
done

NUGET_PACKAGES=~/.nuget/packages

cd $DM_OUT

# Forgot if I needed more than OnnxMl?
dotnet $CSC OnnxMl.cs \
  -r:"$NUGET_PACKAGES/google.protobuf/3.15.0/lib/netstandard2.0/Google.Protobuf.dll" \
  -r:"$DOTNET_CORE/System.Runtime.dll" \
  -r:"$DOTNET_CORE/netstandard.dll"  \
  -r:"$DOTNET_CORE/System.Private.CoreLib.dll" \
  -r:"$DOTNET_CORE/System.Collections.dll" \
  -target:library \
  -out:"./OnnxMlProto.dll"

