#!bash/sh

git submodule update --init --recursive # to get onnx protobuf files

#for f in /usr/share/dotnet/* ; do sudo ln -s $f /usr/lib/dotnet ; done

CSC=/usr/share/dotnet/sdk/7.0.306/Roslyn/bincore/csc.dll

pip3 install onnx # version 1.14

protos=("onnx-ml")

DM_OUT=tmp

mkdir -p $DM_OUT

for str in ${protos[@]}; do
  protoc --csharp_out=$DM_OUT/ -Isubmodules/onnx onnx/$str.proto3
done

{
    echo 'using System;'
    echo 'using System.Reflection;'
    echo '[assembly: System.Reflection.AssemblyCompanyAttribute("OnnxMlProto")]'
    echo '[assembly: System.Reflection.AssemblyConfigurationAttribute("Debug")]'
    echo '[assembly: System.Reflection.AssemblyFileVersionAttribute("1.0.0.0")]'
    echo '[assembly: System.Reflection.AssemblyInformationalVersionAttribute("1.0.0")]'
    echo '[assembly: System.Reflection.AssemblyProductAttribute("OnnxMlProto")]'
    echo '[assembly: System.Reflection.AssemblyTitleAttribute("OnnxMlProto")]'
    echo '[assembly: System.Reflection.AssemblyVersionAttribute("1.0.0.0")]'
    echo '[assembly: global::System.Runtime.Versioning.TargetFrameworkAttribute(".NETStandard,Version=v2.0", FrameworkDisplayName = ".NET Standard 2.0")]'
} > Attributes.cs

NUGET_PACKAGES=~/.nuget/packages
NETSTD=$NUGET_PACKAGES/netstandard.library/2.0.3/build/netstandard2.0/ref/

NET_STANDARD_CMD=$(echo -e "/unsafe-" \
       "/checked-" \
       "/nowarn:1701,1702,IL2121,1701,1702,2008" \
       "/fullpaths" \
       "/nostdlib+" \
       "/errorreport:prompt" \
       "/define:TRACE;DEBUG;NETSTANDARD;NETSTANDARD2_0;NETSTANDARD1_0_OR_GREATER;NETSTANDARD1_1_OR_GREATER;NETSTANDARD1_2_OR_GREATER;NETSTANDARD1_3_OR_GREATER;NETSTANDARD1_4_OR_GREATER;NETSTANDARD1_5_OR_GREATER;NETSTANDARD1_6_OR_GREATER;NETSTANDARD2_0_OR_GREATER" \
       "/errorendlocation" \
       "/preferreduilang:en-US" \
       "/highentropyva+" \
       "/debug+" \
       "/debug:portable" \
       "/filealign:512" \
       "/optimize-" \
       "/out:OnnxMlProto.dll" \
       "/target:library" \
       "/warnaserror-" \
       "/utf8output" \
       "/deterministic+" \
       "/langversion:7.3" \
       "/warnaserror+:NU1605" \
       $(ls $NETSTD | grep \.dll$ | while read line; do echo "/reference:$NETSTD$line";done | tr '\n' ' ' ) \
       "/reference:$NUGET_PACKAGES/google.protobuf/3.23.4/lib/netstandard2.0/Google.Protobuf.dll" \
       "Attributes.cs" \
       "tmp/OnnxMl.cs" 
)

dotnet $CSC $NET_STANDARD_CMD
