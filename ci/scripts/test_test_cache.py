#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ci/scripts/test_cache.py.

Runs against synthetic config roots and build directories, so the expectations
do not drift as the real CI configuration changes.

The load-bearing group is test_keyed_env: the fingerprint hashes only
CONFIG_PATHS rather than the whole ci/ tree, which is sound only because every
job variable that can change a test's outcome is keyed by value.  Those checks
pin that claim, so widening the YAML's influence again has to be deliberate.

Usage: python3 ci/scripts/test_test_cache.py
"""

import json
import os
import shutil
import stat
import struct
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_cache
from test_cache import fingerprint, read_manifest, test_key, write_manifest

FAILURES = []


def check(condition, message):
    if condition:
        return
    FAILURES.append(message)
    print("FAIL: %s" % message)


# Enough of a config root to exercise both CONFIG_PATHS entries plus the
# orchestration files that must no longer count.
CONFIG_FIXTURE = {
    "ci/scripts/test.linux.script.sh": "#!/bin/bash\nctest\n",
    "ci/scripts/common.linux.before_script.sh": "#!/bin/bash\nexport NPROC=`nproc`\n",
    "ci/docker/ubuntu-24.04-amd64-smoketest-run/Dockerfile": "FROM ubuntu:24.04\n",
    "ci/test.linux.gitlab-ci.yml": "test:linux:x86-64:\n  tags: [saas-linux-2xlarge-amd64]\n",
    "ci/common.gitlab-ci.yml": ".rules:libeigen:all-tests:\n  rules: []\n",
    "ci/CTest2JUnit.xsl": "<xsl:stylesheet/>\n",
    "ci/README.md": "CI documentation.\n",
    ".gitlab-ci.yml": "stages: [checkformat, build, test]\n",
}


def write_tree(root, files):
    for rel, content in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(content)


def fingerprint_of(root, env=None):
    """The fingerprint with a controlled environment.  file_digest is cached on
    the path, so an in-place edit is invisible without a cache_clear."""
    test_cache.file_digest.cache_clear()
    saved = os.environ.copy()
    os.environ.clear()
    os.environ.update({"CI_JOB_IMAGE": "ubuntu:24.04"})
    if env:
        os.environ.update(env)
    try:
        return fingerprint(root)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def edit(root, rel, content):
    with open(os.path.join(root, rel), "w") as handle:
        handle.write(content)


def test_config_paths(root):
    """ci/scripts and ci/docker invalidate; the CI YAML no longer does."""
    base = fingerprint_of(root)

    edit(root, "ci/scripts/test.linux.script.sh", "#!/bin/bash\nctest --verbose\n")
    check(fingerprint_of(root) != base, "editing ci/scripts/ must change the fingerprint")
    edit(root, "ci/scripts/test.linux.script.sh", CONFIG_FIXTURE["ci/scripts/test.linux.script.sh"])
    check(fingerprint_of(root) == base, "restoring ci/scripts/ must restore the fingerprint")

    edit(root, "ci/docker/ubuntu-24.04-amd64-smoketest-run/Dockerfile", "FROM ubuntu:26.04\n")
    check(fingerprint_of(root) != base, "editing ci/docker/ must change the fingerprint")
    edit(root, "ci/docker/ubuntu-24.04-amd64-smoketest-run/Dockerfile",
         CONFIG_FIXTURE["ci/docker/ubuntu-24.04-amd64-smoketest-run/Dockerfile"])

    # A new script must not be invisible: this is why CONFIG_PATHS is walked
    # rather than enumerated.
    added = os.path.join(root, "ci/scripts/brand_new.sh")
    with open(added, "w") as handle:
        handle.write("#!/bin/bash\n")
    check(fingerprint_of(root) != base, "adding a file under ci/scripts/ must change the fingerprint")
    os.remove(added)

    # Orchestration: must not invalidate.
    for rel, changed in (
        ("ci/test.linux.gitlab-ci.yml", "test:linux:x86-64:\n  tags: [saas-linux-large-amd64]\n"),
        ("ci/common.gitlab-ci.yml", ".rules:libeigen:all-tests:\n  rules: [{if: $CI_COMMIT_TAG}]\n"),
        ("ci/CTest2JUnit.xsl", "<xsl:stylesheet version='1.0'/>\n"),
        ("ci/README.md", "Rewritten CI documentation.\n"),
        (".gitlab-ci.yml", "stages: [checkformat, build, test, deploy]\n"),
    ):
        edit(root, rel, changed)
        check(fingerprint_of(root) == base, "editing %s must not change the fingerprint" % rel)
        edit(root, rel, CONFIG_FIXTURE[rel])


def test_keyed_env(root):
    """Every knob the old whole-ci/ hash was justified by is keyed by value.

    This is what makes dropping the YAML from the hash sound; if one of these
    regresses, the narrowing is no longer safe.
    """
    base = fingerprint_of(root)
    for name, value in (
        ("EIGEN_REPEAT", "2"),
        ("EIGEN_SEED", "12345"),
        ("EIGEN_CI_CTEST_ARGS", "--timeout 3000"),
        ("QEMU_CPU", "max,sme=on,sme512=on"),
        ("ASAN_OPTIONS", "detect_leaks=1"),
        ("UBSAN_OPTIONS", "print_stacktrace=1"),
        ("LSAN_OPTIONS", "suppressions=lsan.supp"),
        ("TSAN_OPTIONS", "halt_on_error=1"),
        ("MSAN_OPTIONS", "poison_in_dtor=1"),
        ("LD_LIBRARY_PATH", "/usr/aarch64-linux-gnu/lib"),
        ("LD_PRELOAD", "libasan.so"),
    ):
        check(fingerprint_of(root, {name: value}) != base,
              "%s must be part of the fingerprint" % name)

    # The image is keyed; host-derived values that vary across runners of one
    # job are deliberately not.
    check(fingerprint_of(root, {"CI_JOB_IMAGE": "ubuntu:26.04"}) != base,
          "CI_JOB_IMAGE must be part of the fingerprint")
    for name, value in (("EIGEN_CI_CTEST_PARALLEL", "4"), ("NPROC", "32"), ("CI_JOB_ID", "12345")):
        check(fingerprint_of(root, {name: value}) == base,
              "%s must not be part of the fingerprint" % name)


def test_fingerprint_stable(root):
    check(fingerprint_of(root) == fingerprint_of(root),
          "the fingerprint must not depend on walk or dict order")


def make_test(builddir, name="unit", command=None, properties=None):
    binary = os.path.join(builddir, name)
    if not os.path.exists(binary):
        with open(binary, "w") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
    return {"name": name, "command": command or [binary], "properties": properties or []}


def test_keying(builddir):
    fp = b"fingerprint\0"
    unit = make_test(builddir)
    base = test_key(unit, fp, builddir)
    check(base is not None, "a command ending in a build-directory executable must be keyable")

    # The binary's content is what the cache is addressed by.
    test_cache.file_digest.cache_clear()
    with open(os.path.join(builddir, "unit"), "w") as handle:
        handle.write("#!/bin/sh\nexit 1\n")
    check(test_key(unit, fp, builddir) != base, "changing the binary must change the key")
    test_cache.file_digest.cache_clear()
    with open(os.path.join(builddir, "unit"), "w") as handle:
        handle.write("#!/bin/sh\nexit 0\n")
    test_cache.file_digest.cache_clear()
    check(test_key(unit, fp, builddir) == base, "restoring the binary must restore the key")

    # A non-final command element that is a file -- the emulator on a
    # cross-compiled job -- is keyed by content, not by name.
    emulator = os.path.join(builddir, "qemu-aarch64-static")
    with open(emulator, "w") as handle:
        handle.write("emulator v1\n")
    emulated = make_test(builddir, command=[emulator, os.path.join(builddir, "unit")])
    test_cache.file_digest.cache_clear()
    with_v1 = test_key(emulated, fp, builddir)
    with open(emulator, "w") as handle:
        handle.write("emulator v2\n")
    test_cache.file_digest.cache_clear()
    check(test_key(emulated, fp, builddir) != with_v1, "the emulator's content must be part of the key")

    # CTest properties reach the key, which is how --timeout is covered twice.
    timed = make_test(builddir, properties=[{"name": "TIMEOUT", "value": "1500"}])
    other = make_test(builddir, properties=[{"name": "TIMEOUT", "value": "3000"}])
    test_cache.file_digest.cache_clear()
    check(test_key(timed, fp, builddir) != test_key(other, fp, builddir),
          "a CTest property must be part of the key")

    # The fingerprint reaches the key.
    test_cache.file_digest.cache_clear()
    check(test_key(unit, b"other\0", builddir) != base, "the fingerprint must reach the test key")

    # Not keyable: the outcome depends on more than the artifact's content.
    check(test_key({"name": "n", "command": []}, fp, builddir) is None, "an empty command is not keyable")
    check(test_key({"name": "n"}, fp, builddir) is None, "a missing command is not keyable")
    check(test_key({"name": "n", "command": ["/bin/sh", "-c", "true"]}, fp, builddir) is None,
          "a command outside the build directory is not keyable")


def clear_caches():
    test_cache.file_digest.cache_clear()
    test_cache.needed_libraries.cache_clear()
    test_cache.build_tree_files.cache_clear()


def write_elf(path, needed=None, is64=True, big_endian=False):
    """A minimal ELF shared object whose dynamic section lists `needed`, or
    one without a PT_DYNAMIC segment, like a static executable, if `needed`
    is None.  The PT_LOAD segment maps the file at a nonzero address, so
    reading DT_STRTAB needs the address-to-offset translation."""
    order = ">" if big_endian else "<"
    word = "Q" if is64 else "I"
    entry = order + ("qQ" if is64 else "iI")
    ehsize, phentsize = (64, 56) if is64 else (52, 32)
    base = 0x400000
    strtab = b"\0"
    tags = []
    for name in needed or ():
        tags.append((test_cache.DT_NEEDED, len(strtab)))
        strtab += name.encode() + b"\0"
    phnum = 2 if needed is not None else 1
    dynoff = ehsize + phnum * phentsize
    stroff = dynoff + (len(tags) + 2) * struct.calcsize(entry)
    tags += [(test_cache.DT_STRTAB, base + stroff), (test_cache.DT_NULL, 0)]
    dynsize = stroff - dynoff
    filesize = stroff + len(strtab)

    def phdr(p_type, offset, size):
        if is64:
            return struct.pack(order + "IIQQQQQQ", p_type, 4, offset, base + offset, base + offset, size, size, 8)
        return struct.pack(order + "IIIIIIII", p_type, offset, base + offset, base + offset, size, size, 4, 4)

    ident = b"\x7fELF" + bytes([2 if is64 else 1, 2 if big_endian else 1, 1]) + bytes(9)
    header = ident + struct.pack(order + "HHI" + word * 3 + "IHHHHHH", 3, 0, 1, 0, ehsize, 0, 0, ehsize,
                                 phentsize, phnum, 0, 0, 0)
    data = header + phdr(test_cache.PT_LOAD, 0, filesize)
    if needed is not None:
        data += phdr(test_cache.PT_DYNAMIC, dynoff, dynsize)
        data += b"".join(struct.pack(entry, tag, value) for tag, value in tags) + strtab
    with open(path, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o755)


def test_needed_libraries(builddir):
    """DT_NEEDED is read through the program headers for both ELF classes and
    byte orders; anything else has no libraries, and a truncated ELF file
    raises so that `plan` fails closed."""
    path = os.path.join(builddir, "elf")
    for is64, big_endian in ((True, False), (False, True)):
        write_elf(path, ["libfoo.so.1", "libc.so.6"], is64, big_endian)
        clear_caches()
        check(test_cache.needed_libraries(path) == ("libfoo.so.1", "libc.so.6"),
              "DT_NEEDED must be read from a %d-bit %s-endian ELF file, got %s"
              % (64 if is64 else 32, "big" if big_endian else "little", test_cache.needed_libraries(path)))

    # Linker output has the layouts write_elf does not: several PT_LOAD
    # segments, .dynstr outside the first.  The interpreter running this
    # links its C library: libc.so.6 (glibc) or libc.musl-<arch>.so.1 (musl,
    # the lint image).  Not ELF, as on macOS: nothing to check.
    interpreter = os.path.realpath(sys.executable)
    with open(interpreter, "rb") as handle:
        is_elf = handle.read(4) == b"\x7fELF"
    if is_elf:
        clear_caches()
        names = test_cache.needed_libraries(interpreter)
        check(any(name.startswith("libc.") for name in names),
              "DT_NEEDED of %s must name its C library, got %s" % (interpreter, names))

    write_elf(path)
    clear_caches()
    check(test_cache.needed_libraries(path) == (), "an ELF file without PT_DYNAMIC loads no libraries")

    with open(path, "w") as handle:
        handle.write("#!/bin/sh\nexit 0\n")
    clear_caches()
    check(test_cache.needed_libraries(path) == (), "a file that is not ELF loads no libraries")

    write_elf(path, ["libfoo.so.1"])
    with open(path, "r+b") as handle:
        handle.truncate(70)
    clear_caches()
    try:
        test_cache.needed_libraries(path)
    except (struct.error, ValueError):
        pass
    else:
        check(False, "a truncated ELF file must raise")
    os.remove(path)


def test_build_tree_libraries(builddir):
    """A test's key covers the build-directory libraries it loads, directly or
    through another one, and no others."""
    for sub in ("test", "lib", "other"):
        os.makedirs(os.path.join(builddir, sub), exist_ok=True)
    binary = os.path.join(builddir, "test", "unit")
    foo = os.path.join(builddir, "lib", "libfoo.so.1.0")
    bar = os.path.join(builddir, "other", "libbar.so")
    unrelated = os.path.join(builddir, "lib", "libunrelated.so")
    libraries = {foo: ["libbar.so"], bar: [], unrelated: []}
    write_elf(binary, ["libfoo.so.1", "libc.so.6"])
    for library, needed in libraries.items():
        write_elf(library, needed)
    os.symlink("libfoo.so.1.0", os.path.join(builddir, "lib", "libfoo.so.1"))

    clear_caches()
    check(test_cache.build_tree_libraries([binary], builddir) == sorted([foo, bar]),
          "the build-tree libraries must be found by DT_NEEDED name, transitively, got %s"
          % test_cache.build_tree_libraries([binary], builddir))

    unit = {"name": "unit", "command": [binary], "properties": []}
    fp = b"fingerprint\0"
    base = test_key(unit, fp, builddir)
    for library, changes in ((foo, True), (bar, True), (unrelated, False)):
        with open(library, "ab") as handle:
            handle.write(b"rebuilt")
        clear_caches()
        changed = test_key(unit, fp, builddir) != base
        check(changed == changes, "changing %s must %schange the key"
              % (os.path.basename(library), "" if changes else "not "))
        write_elf(library, libraries[library])
        clear_caches()
        check(test_key(unit, fp, builddir) == base,
              "restoring %s must restore the key" % os.path.basename(library))


def test_manifest_roundtrip(builddir):
    path = os.path.join(builddir, "manifest")
    write_manifest(path, {"k1": "a", "k2": "b", "k3": "c"}, 10)
    check(list(read_manifest(path).items()) == [("k1", "a"), ("k2", "b"), ("k3", "c")],
          "the manifest must round-trip in insertion order")

    # Trimming keeps the most-recently-useful entries, which are at the back.
    write_manifest(path, {"k1": "a", "k2": "b", "k3": "c"}, 2)
    check(list(read_manifest(path)) == ["k2", "k3"], "trimming must drop the oldest entries")

    check(read_manifest(os.path.join(builddir, "absent")) == {},
          "a missing manifest must read as empty")


def test_record_status(builddir):
    """Only first-attempt passes enter the manifest, and an unreadable
    Test.xml records nothing rather than trusting a partial run."""

    class Args(object):
        pass

    testing = os.path.join(builddir, "Testing")
    os.makedirs(os.path.join(testing, "20260825-0000"), exist_ok=True)
    with open(os.path.join(testing, "TAG"), "w") as handle:
        handle.write("20260825-0000\n")

    root = ElementTree.Element("Site")
    testing_el = ElementTree.SubElement(root, "Testing")
    for name, status in (("ok", "passed"), ("bad", "failed"), ("skipped", "notrun")):
        el = ElementTree.SubElement(testing_el, "Test", Status=status)
        ElementTree.SubElement(el, "Name").text = name
    xml_path = os.path.join(testing, "20260825-0000", "Test.xml")
    ElementTree.ElementTree(root).write(xml_path)

    keys_path = os.path.join(builddir, "keys")
    with open(keys_path, "w") as handle:
        handle.write("key-ok ok\nkey-bad bad\nkey-skipped skipped\n")

    args = Args()
    args.manifest = os.path.join(builddir, "record-manifest")
    args.keys = keys_path
    args.testing_dir = testing
    args.max_entries = 20000
    if os.path.exists(args.manifest):
        os.remove(args.manifest)
    test_cache.record(args)
    recorded = read_manifest(args.manifest)
    check(list(recorded) == ["key-ok"],
          "only Status=passed may be recorded, got %s" % list(recorded))

    # A truncated Test.xml must leave an existing manifest untouched.
    write_manifest(args.manifest, {"key-existing": "existing"}, 10)
    with open(xml_path, "w") as handle:
        handle.write("<Site><Testing><Test Status=")
    test_cache.record(args)
    check(list(read_manifest(args.manifest)) == ["key-existing"],
          "an unparsable Test.xml must record nothing")

    os.remove(os.path.join(testing, "TAG"))
    test_cache.record(args)
    check(list(read_manifest(args.manifest)) == ["key-existing"],
          "a missing TAG must record nothing")


def main():
    root = tempfile.mkdtemp(prefix="eigen-test-cache-")
    builddir = tempfile.mkdtemp(prefix="eigen-test-cache-build-")
    try:
        write_tree(root, CONFIG_FIXTURE)
        test_config_paths(root)
        test_keyed_env(root)
        test_fingerprint_stable(root)
        test_keying(os.path.realpath(builddir))
        libs = os.path.join(os.path.realpath(builddir), "libs")
        os.makedirs(libs)
        test_needed_libraries(libs)
        test_build_tree_libraries(libs)
        test_manifest_roundtrip(builddir)
        test_record_status(builddir)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(builddir, ignore_errors=True)

    if FAILURES:
        print("\n%d check(s) failed" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
